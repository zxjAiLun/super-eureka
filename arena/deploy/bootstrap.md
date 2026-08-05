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

### Deploy user (for GitHub Actions)

The workflows SSH into the server as a non-root `deploy` user. Create it with a
minimal sudoers entry (P1.7): the workflows do all file operations as
`chessarena` (no root), and only restart the two services as root.

```bash
sudo useradd --create-home --shell /bin/bash deploy
sudo mkdir -p /opt/chessarena/incoming
# Shared group: deploy uploads into incoming/, chessarena extracts from it.
sudo chown deploy:chessarena /opt/chessarena/incoming
sudo chmod 775 /opt/chessarena/incoming

# Minimal sudoers:
#   - run anything as chessarena (all file/venv/migration work),
#   - restart the two arena services only.
cat > /etc/sudoers.d/chessarena-deploy <<'EOF'
deploy ALL=(chessarena) NOPASSWD: ALL
deploy ALL=(root) NOPASSWD: /usr/bin/systemctl restart chessarena-api
deploy ALL=(root) NOPASSWD: /usr/bin/systemctl restart chessarena-worker
deploy ALL=(root) NOPASSWD: /usr/bin/systemctl is-active chessarena-*
EOF
sudo chmod 0440 /etc/sudoers.d/chessarena-deploy
```

Verify with `sudo -l -U deploy` (as root) that the `deploy` user can run
`sudo -u chessarena ...` and `sudo systemctl restart chessarena-api` without a
password.

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

## 9. GitHub Actions -> server SSH access

The workflows SSH from GitHub Actions **into** the server, so you need a
keypair where:

- the **private** key is stored in the GitHub repository secrets as
  `ARENA_DEPLOY_KEY`, and
- the **public** key is installed in the `deploy` user's `authorized_keys`.

Generate the keypair on your workstation (NOT on the server):

```bash
ssh-keygen -t ed25519 -f ./arena_deploy_ed25519 -N "" -C "chessarena-deploy"
cat ./arena_deploy_ed25519.pub
```

Then on the server, install the public key for the `deploy` user:

```bash
sudo -u deploy mkdir -p /home/deploy/.ssh
sudo -u deploy sh -c 'echo "<paste the public key>" >> /home/deploy/.ssh/authorized_keys'
sudo -u deploy chmod 700 /home/deploy/.ssh
sudo -u deploy chmod 600 /home/deploy/.ssh/authorized_keys
```

Add the private key and host metadata as repository secrets:

```text
ARENA_DEPLOY_KEY        <contents of ./arena_deploy_ed25519>
ARENA_DEPLOY_HOST       pearllover.site
ARENA_DEPLOY_USER       deploy
ARENA_SERVER_HOST_KEY   <output of: ssh-keyscan pearllover.site>
```

Note on GitHub "deploy keys": those are for letting the **server** clone the
repository, which this setup does not need. The workflows upload artifacts via
rsync/scp; the reverse direction (Actions -> server) is configured above.

Verify the connection before running the workflows:

```bash
ssh -i ./arena_deploy_ed25519 deploy@pearllover.site \
  'sudo systemctl is-active chessarena-api chessarena-worker'
```

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
