# Rootless Podman Deployment Guide

This guide covers deploying the GSRS MCP Server with `podman kube play` in
**rootless** mode, including the host firewall port-forwards that map the
rootless container host ports (`8080` / `8443`) onto the well-known HTTP/HTTPS
ports (`80` / `443`).

> For the manifest itself see [`podman-kube-play.yaml`](../../podman-kube-play.yaml)
> in the repository root. For the equivalent non-rootless / docker-compose
> deployment see [`docker-compose.yaml`](../../docker-compose.yaml).

## Why rootless?

Rootless podman runs containers as an unprivileged user (your normal login
account) with a user namespace mapping. This is the recommended setup on
shared / multi-tenant hosts because:

- Containers cannot escalate to real `root` on the host
- The podman API and the user-mode networking stack are isolated from system
  services
- You do not need to grant the user passwordless `sudo` for normal pod
  lifecycle operations (`kube play`, `kube down`, `ps`, `logs`, ...)

The trade-off is that **container ports cannot bind below `1024` on the
host** (the "privileged port" range) — the pod's `hostPort:` values are
remapped into the unprivileged range. In the GSRS manifest, the caddy
container exposes:

| Container port | Host port (rootless) |
| -------------- | -------------------- |
| `80`           | `8080`               |
| `443`          | `8443`               |

The mcp-server container's `8000` port is bound directly to the host
(`hostPort: 8000`) and is not affected by the privileged-port restriction
because the mcp-server is reached through caddy, not directly from outside.

## Architecture

```
                    Internet
                       │
                  ┌────┴────┐
                  │  host   │  ports 80, 443 (firewall-cmd forwards)
                  └────┬────┘
                       │
       ┌───────────────┴───────────────┐
       │ firewalld (public zone)       │
       │  80  →  host:8080             │
       │  443 →  host:8443             │
       └───────────────┬───────────────┘
                       │
                  ┌────┴────┐
                  │  podman │  rootlessport
                  └────┬────┘
                       │
              gsrs-mcp pod
              ┌──────────────┐
              │   httpd      │  caddy:2  :80/:443
              │  (caddy)     │  hostPort 8080/8443
              ├──────────────┤
              │   server     │  gsrs-mcp-server  :8000
              │  (mcp)       │  hostPort 8000
              ├──────────────┤
              │   vectordb   │  pgvector  :5432 (pod-internal)
              └──────────────┘
```

## Host firewall configuration

The host runs `firewalld` (default on Fedora / RHEL / Rocky). The rootless
pod publishes caddy on `hostPort 8080` / `8443` instead of `80` / `443`, so
external clients (browsers, MCP clients) cannot reach the service on the
standard ports until the host is configured to forward them.

The required rules, in the order they are typically applied:

```bash
# 1. Forward the well-known HTTP/HTTPS ports to the rootless
#    container host ports.
sudo firewall-cmd --add-forward-port=port=80:proto=tcp:toport=8080 --permanent
sudo firewall-cmd --add-forward-port=port=443:proto=tcp:toport=8443 --permanent

# 2. Open the http and https services in the public zone.
sudo firewall-cmd --permanent --zone=public --add-service=https
sudo firewall-cmd --permanent --zone=public --add-service=http

# 3. (Optional, defensive) Also open the bare ports explicitly so the
#    service rules cannot be removed by a later `firewall-cmd` reload
#    that drops the service definitions.
sudo firewall-cmd --permanent --zone=public --add-port=443/tcp
sudo firewall-cmd --permanent --zone=public --add-port=80/tcp

# 4. Apply the new runtime state. `--permanent` only writes the
#    on-disk config; the running firewall still needs a reload.
sudo firewall-cmd --reload
```

Verify the result:

```bash
sudo firewall-cmd --list-all --zone=public
```

You should see, among other lines:

```
services: ... http https ...
ports: ... 80/tcp 443/tcp ...
forward-ports:
  port=80:proto=tcp:toport=8080:toaddr=
  port=443:proto=tcp:toport=8443:toaddr=
```

> The `forward-port` rules are what route the public-facing
> `80` / `443` traffic into the rootless `8080` / `8443` bound by caddy.

## Confirm rootless is enabled

These commands do not require root and should succeed:

```bash
podman info | grep -E "rootless|rootlessport"
# rootless: true
# rootlessport: true  (or slirp4netns on older setups)
```

If `rootless` is `false` you are running the system podman instance. Either
log out and back in (rootless is the default when the user is in the
`wheel` group and the user session is fresh) or use the `--runroot` /
`--root` flags of `podman` to point at your user's store explicitly.

## Deploy

```bash
# 1. Build the app image (one-time, or after code changes)
podman build -t localhost/gsrs-mcp-server:latest .

# 2. Generate the ConfigMap file from your project .env.
#    `mcp-config.yaml` carries three `kind: ConfigMap` resources
#    that the pod's containers pull from via
#    `envFrom: configMapRef:`:
#      * `vectordb-env`    — postgres / DATABASE_URL keys
#      * `server-env`      — EMBEDDING_*, MCP_*, GSRS_API_*, CHUNKER_*,
#                             DEFAULT_TOP_K, DEBUG_MODE, etc.
#      * `httpd-env`       — MCP_HOSTNAME, MCP_UPSTREAM, CADDY_*
#
#    All three ConfigMaps are produced automatically from `.env`
#    by a one-line awk that emits one `data:` block per key. Any
#    tool that emits the right YAML works (yq, envsubst + a
#    template, etc.).
#
#    A simple awk-based generator (cross-platform, no deps):
awk '
  BEGIN {
    print "# Generated from .env — do not edit by hand."
    print "apiVersion: v1"
    print "kind: ConfigMap"
    print "metadata:"
    print "  name: vectordb-env"
    print "data:"
  }
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  {
    n = index($0, "=")
    if (n == 0) next
    key = substr($0, 1, n - 1)
    val = substr($0, n + 1)
    # YAML double-quote escaping: backslash and double-quote.
    gsub(/\\/, "\\\\", val); gsub(/"/, "\\\"", val)
    printf("  %s: \"%s\"\n", key, val)
  }
' .env > mcp-config.yaml

# 3. Apply both files: the ConfigMaps are loaded via the
#    `--configmap` flag, the Pod via the positional argument.
podman kube play --configmap mcp-config.yaml podman-kube-play.yaml

# 4. Check that all three containers came up
podman ps --filter "name=gsrs-mcp" --format "{{.Names}}\t{{.Status}}"
# gsrs-mcp-vectordb  Up 5 seconds
# gsrs-mcp-server    Up 5 seconds
# gsrs-mcp-httpd     Up 5 seconds
```

> **What changed:** the manifest no longer carries any env values
> or `${KEY}` placeholders. All env values live in
> [`mcp-config.yaml`](../../mcp-config.yaml) (a sibling file,
> gitignored) and are passed to `podman kube play` via the
> `--configmap` flag. The three containers each pull their own
> scoped ConfigMap via `envFrom: configMapRef: { name: <cm> }`
> instead of carrying their own inline `env:` list. This avoids
> the long-standing envsubst-vs-caddy incompatibility (envsubst
> mangled the Caddyfile's `{$VAR:default}` placeholders because
> it sees `${VAR:default}` inside the Caddyfile and substitutes
> the var name, leaving the `:` and default). The Caddyfile's
> `{$VAR:default}` placeholders are still useful as a final
> safety net for keys the operator chooses not to put in `.env`.

### How the ConfigMap flow works

`podman kube play --configmap <file> <manifest>` loads every
`kind: ConfigMap` resource from `<file>` into the pod and uses
them as the env source for any container that references them by
name. The flag is documented in `podman-kube-play(1)` and accepts
multi-doc YAML. The three ConfigMaps in `mcp-config.yaml` are:

| ConfigMap name    | Consumed by      | Purpose                                                |
| ----------------- | ---------------- | ------------------------------------------------------ |
| `vectordb-env`    | `vectordb`       | `POSTGRES_*`, `DATABASE_URL`                            |
| `server-env`      | `server`         | `EMBEDDING_*`, `MCP_*`, `GSRS_API_*`, `CHUNKER_*`, etc. |
| `httpd-env`       | `httpd` (caddy)  | `MCP_HOSTNAME`, `MCP_UPSTREAM`, `CADDY_*`              |

The caddy container's Caddyfile's `{$VAR:default}` placeholders
resolve against the caddy container's own env (which now has real
values, not literal `${VAR}` strings).

### Overriding the caddy defaults

The caddy container ships with these safe-but-useless defaults baked
into the Caddyfile's `{$VAR:default}` placeholders:

| Env var          | Default                                          |
| ---------------- | ------------------------------------------------ |
| `MCP_HOSTNAME`   | `gsrs.example.com`                               |
| `MCP_UPSTREAM`   | `localhost:8000` (the in-pod mcp-server)         |
| `CADDY_ACME_CA`  | `https://acme-v02.api.letsencrypt.org/directory` |
| `CADDY_EMAIL`    | `admin@example.com`                              |

To use a different hostname, internal CA, or contact email, set the
matching key in `.env` (e.g. `MCP_HOSTNAME=foo.example.com`) and
re-render the manifest with one of the two workflows above. Caddy
reads the resolved env at container startup.

## Verify end-to-end

From the host (and from another machine on the internet, if the host's
firewall public zone permits it):

```bash
curl -kI https://localhost/livez
# HTTP/2 200
curl -kI https://localhost/readyz
# HTTP/2 200
```

`-k` is required when `CADDY_ACME_CA` points at an internal /
private CA whose root is not in the host's system trust store. To
make `curl` trust it without `-k`, copy the CA chain to
`/etc/pki/ca-trust/source/anchors/` and run `update-ca-trust` (or
the equivalent for your distro). In a browser, install the CA
chain and load `https://<MCP_HOSTNAME>/` to see the caddy-served
UI / proxied MCP endpoints.

## Tear down

```bash
# Stop the pod and remove the volumes it created
podman kube down podman-kube-play.yaml
```

The `firewalld` rules are **persistent** and survive `kube down`. Remove
them explicitly if you want a clean state:

```bash
sudo firewall-cmd --remove-forward-port=port=80:proto=tcp:toport=8080 --permanent
sudo firewall-cmd --remove-forward-port=port=443:proto=tcp:toport=8443 --permanent
sudo firewall-cmd --permanent --zone=public --remove-service=https
sudo firewall-cmd --permanent --zone=public --remove-service=http
sudo firewall-cmd --permanent --zone=public --remove-port=443/tcp
sudo firewall-cmd --permanent --zone=public --remove-port=80/tcp
sudo firewall-cmd --reload
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `connection refused` on `https://<host>/` from outside | The firewall `forward-port` rules were added with `--permanent` but not reloaded, so the **runtime** firewall does not have them yet. | `sudo firewall-cmd --reload` |
| `connection refused` on `https://<host>/` from the host itself | Rootlessport uses hairpin NAT, but only some network stacks do. On older podman / slirp4netns, hitting `127.0.0.1` may not route back to the container. | Hit the pod's own host IP (`hostname -i`) or the pod-internal `gsrs-mcp-httpd` IP, or upgrade to netavark (default since podman 4.x). |
| `no route to host` from another machine on the LAN | The `public` zone is bound to the WAN interface only, not the LAN. | Add the LAN interface to the `public` zone (or use `--zone=trusted` for the LAN) and reload. |
| caddy logs `parsing key: invalid port '<hostname>}'` (e.g. `'gsrs.example.com}'`) | The caddy container saw a literal `${MCP_HOSTNAME}` instead of a real value. The ConfigMap was not loaded via `--configmap` (or the file you passed is empty or malformed). | Re-generate `mcp-config.yaml` from `.env` with the awk generator in the `## Deploy` section, then re-apply with `podman kube play --configmap mcp-config.yaml podman-kube-play.yaml`. |
| caddy starts but TLS obtain fails with `x509: certificate signed by unknown authority` | `CADDY_ACME_CA` points at an internal / private ACME server whose root is not in the caddy image's trust store. | Either switch `CADDY_ACME_CA` to a publicly-trusted ACME server (e.g. Let's Encrypt), or mount the internal CA chain into the caddy container's trust store. |
| `podman kube play` fails with `error loading configmap ... not found` (or no env vars are injected into the containers) | You forgot to pass `--configmap mcp-config.yaml`. The pod's containers reference `envFrom: configMapRef:` for three ConfigMaps, but they live in a separate file. | Re-run with both files: `podman kube play --configmap mcp-config.yaml podman-kube-play.yaml`. |
| caddy container has no `MCP_HOSTNAME` / `MCP_UPSTREAM` / `CADDY_*` env vars at all | The `--configmap` flag was not passed, OR the file is empty, OR the ConfigMaps inside the file have wrong names (must be `vectordb-env`, `server-env`, `httpd-env`). | Check that `mcp-config.yaml` exists, contains those three `kind: ConfigMap` resources with those exact `metadata.name` values, and is passed via `--configmap`. |
