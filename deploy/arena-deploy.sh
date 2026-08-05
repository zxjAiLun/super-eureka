#!/usr/bin/env bash
# Root-owned deployment wrapper for ChessArena (P1: minimal sudoers).
#
# This is the ONLY command the deploy user is allowed to run through sudo.
# Every argument is validated strictly (release/build id format, no extra
# arguments, realpath confined to the arena trees), so no arbitrary path or
# additional argument can be injected.  The sudoers entry is:
#
#   deploy ALL=(root) NOPASSWD: /opt/chessarena/bin/arena-deploy release-install *
#   deploy ALL=(root) NOPASSWD: /opt/chessarena/bin/arena-deploy release-switch *
#   deploy ALL=(root) NOPASSWD: /opt/chessarena/bin/arena-deploy build-install *
#   deploy ALL=(root) NOPASSWD: /opt/chessarena/bin/arena-deploy restart-api
#   deploy ALL=(root) NOPASSWD: /opt/chessarena/bin/arena-deploy restart-worker
#
# Usage:
#   arena-deploy release-install <YYYYMMDDHHMMSS>
#   arena-deploy release-switch <YYYYMMDDHHMMSS>
#   arena-deploy build-install <YYYYMMDD-hhhhhhh-label>
#   arena-deploy restart-api
#   arena-deploy restart-worker
set -euo pipefail

INCOMING=/opt/chessarena/incoming
RELEASES=/opt/chessarena/releases
BUILDS=/opt/chessarena/builds
CURRENT=/opt/chessarena/app/current
VENV=/opt/chessarena/venv
ENV_FILE=/etc/chessarena/chessarena.env

fail() { echo "arena-deploy: $*" >&2; exit 1; }

release_dir() {
    local id="${1:-}"
    [[ "$id" =~ ^[0-9]{14}$ ]] || fail "invalid release id: '$id'"
    echo "$RELEASES/$id"
}

build_dir() {
    local id="${1:-}"
    [[ "$id" =~ ^[0-9]{8}-[0-9a-f]{7,40}-[A-Za-z0-9._-]+$ ]] || fail "invalid build id: '$id'"
    echo "$BUILDS/$id"
}

case "${1:-}" in
    release-install)
        # Extract the uploaded package into a fresh release dir, install its
        # dependencies and run migrations.  The package is removed from
        # incoming afterwards.
        dest="$(release_dir "${2:-}")"
        [ -d "$dest" ] && fail "release already exists: $dest"
        [ -f "$INCOMING/arena.tar.gz" ] || fail "no package at $INCOMING/arena.tar.gz"
        mkdir -p "$dest"
        tar -xzf "$INCOMING/arena.tar.gz" -C "$dest"
        rm -f "$INCOMING/arena.tar.gz"
        chown -R chessarena:chessarena "$dest"
        sudo -u chessarena "$VENV/bin/pip" install -e "$dest"
        set -a
        # shellcheck disable=SC1091
        . "$ENV_FILE"
        set +a
        sudo -u chessarena "$VENV/bin/alembic" -c "$dest/alembic.ini" upgrade head
        echo "arena-deploy: release installed $dest"
        ;;
    release-switch)
        # Atomically repoint /opt/chessarena/app/current.
        dest="$(release_dir "${2:-}")"
        [ -d "$dest" ] || fail "release not found: $dest"
        ln -sfn "$dest" "$CURRENT"
        echo "arena-deploy: switched current -> $dest"
        ;;
    build-install)
        # Verify the uploaded tarball SHA against its manifest and atomically
        # install it as an immutable build directory, then register it.
        id="${2:-}"
        dest="$(build_dir "$id")"
        tarball="$INCOMING/build-$id.tar.gz"
        [ -f "$tarball" ] || fail "no tarball at $tarball"
        [ -d "$dest" ] && fail "build already exists: $dest"
        stage="$INCOMING/stage-$id"
        rm -rf "$stage"
        mkdir -p "$stage"
        tar -xzf "$tarball" -C "$stage"
        rm -f "$tarball"
        actual=$(sha256sum "$stage/engine/engine" | cut -d' ' -f1)
        expected=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['binary_sha256'])" "$stage/engine/manifest.json")
        [ "$actual" = "$expected" ] || { rm -rf "$stage"; fail "SHA mismatch"; }
        mv "$stage/engine" "$dest"
        rm -rf "$stage"
        chmod 0555 "$dest/engine"
        chmod 0444 "$dest/manifest.json"
        chown -R chessarena:chessarena "$dest"
        set -a
        # shellcheck disable=SC1091
        . "$ENV_FILE"
        set +a
        sudo -u chessarena "$VENV/bin/python" \
            "$CURRENT/scripts/install_build.py" "$dest" --probe
        echo "arena-deploy: build installed $dest"
        ;;
    restart-api)
        systemctl restart chessarena-api
        ;;
    restart-worker)
        systemctl restart chessarena-worker
        ;;
    *)
        fail "usage: arena-deploy {release-install|release-switch|build-install|restart-api|restart-worker} [id]"
        ;;
esac
