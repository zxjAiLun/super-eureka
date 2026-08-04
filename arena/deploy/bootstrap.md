# ChessArena v1 — server bootstrap

One-time procedure (section 20.1). After this completes, day-to-day match
management happens through the web UI and GitHub Actions; no routine SSH is
required.

Requirements: a Linux server (Debian/Ubuntu recommended) with 2 vCPU / 2 GB
RAM / 50 GB SSD, `pearllover.site` pointing at it, and root SSH access for
this one session.

## 1. Install system packages

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip \
    cutechess-cli nginx apache2-utils rsync
```

Verify cutechess:

```bash
cutechess-cli -version
```

## 2. Create the service user and directories

```bash
sudo useradd --system --create-home --home-dir /opt/chessarena chessarena
sudo mkdir -p /opt/chessarena/app /opt/chessarena/builds /opt/chessarena/openings
sudo mkdir -p /var/lib/chessarena/runs /var/lib/chessarena/state
sudo mkdir -p /var/log/chessarena
sudo mkdir -p /etc/chessarena

sudo chown -R chessarena:chessarena /opt/chessarena /var/lib/chessarena /var/log/chessarena
```

The `chessarena` user must NOT have sudo rights. It only needs write access to
`/var/lib/chessarena` and `/opt/chessarena/app`.

## 3. Create the Python virtualenv

```bash
sudo -u chessarena python3.12 -m venv /opt/chessarena/venv
sudo -u chessarena /opt/chessarena/venv/bin/pip install \
    fastapi 'uvicorn[standard]' sqlalchemy alembic jinja2 python-chess pydantic
```

## 4. Configure the environment

```bash
sudo cp chessarena.env /etc/chessarena/chessarena.env
sudo chown root:chessarena /etc/chessarena/chessarena.env
sudo chmod 640 /etc/chessarena/chessarena.env
```

## 5. Deploy the application

The first deployment is performed by the `deploy-arena` GitHub Action, or
manually:

```bash
sudo rsync -a arena/ /opt/chessarena/releases/$(date +%Y%m%d%H%M%S)/
sudo ln -sfn <release-dir> /opt/chessarena/app/current
sudo -u chessarena /opt/chessarena/venv/bin/pip install -e /opt/chessarena/app/current
sudo -u chessarena ARENA_DB_URL=sqlite:////var/lib/chessarena/arena.db \
    /opt/chessarena/venv/bin/alembic -c /opt/chessarena/app/current/alembic.ini upgrade head
```

## 6. Install systemd units

```bash
sudo cp chessarena-api.service chessarena-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chessarena-api chessarena-worker
```

## 7. Configure Nginx

```bash
sudo cp nginx-chessarena.conf /etc/nginx/snippets/chessarena.conf
```

Add `include /etc/nginx/snippets/chessarena.conf;` inside the `443` server
block for `pearllover.site`, then:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 8. Basic Auth

```bash
sudo htpasswd -c /etc/chessarena/.htpasswd arena-admin
sudo chown root:chessarena /etc/chessarena/.htpasswd
sudo chmod 640 /etc/chessarena/.htpasswd
```

## 9. GitHub deploy key

Create a deploy key for the `chessarena` user so GitHub Actions can deploy
without credentials:

```bash
sudo -u chessarena ssh-keygen -t ed25519 -f /opt/chessarena/.ssh/id_ed25519 -N ""
```

Add the public key as a **deploy key** on the repository (read-only is
sufficient for `git` fetch; the workflow pushes artifacts via rsync, so grant
write access to the repo and add the private key to the repository secret
`ARENA_DEPLOY_KEY` plus host key to `ARENA_SERVER_HOST_KEY`).

## 10. Register the initial build and opening set

```bash
sudo -u chessarena /opt/chessarena/venv/bin/python \
    /opt/chessarena/app/current/scripts/install_build.py \
    /opt/chessarena/builds/<build_id> --probe

sudo -u chessarena /opt/chessarena/venv/bin/python \
    /opt/chessarena/app/current/scripts/register_openings.py \
    /opt/chessarena/openings/<opening_set_id>/openings.epd \
    /opt/chessarena/openings/<opening_set_id>/manifest.json
```

## 11. Verify

```bash
sudo -u chessarena /opt/chessarena/venv/bin/python \
    /opt/chessarena/app/current/scripts/verify_install.py
```

Then open https://pearllover.site/chessarena/admin/ and confirm the dashboard
shows the worker online.
