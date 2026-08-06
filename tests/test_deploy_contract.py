"""Deploy-artifact contract tests.

Regression checks on the deployed systemd unit and the GitHub Actions
deployment workflows:

- the API unit must start the application through its factory
  (chessarena.main:create_app --factory).  The API module exposes
  create_app(), not a module-level app, so the obsolete
  chessarena.main:app target crashes the service at bootstrap
  (uvicorn: Attribute "app" not found).
- the deploy workflows must stage their upload archives inside the
  repository workspace under a relative path.  appleboy/scp-action runs in
  a Docker container whose working directory is /github/workspace, so a
  runner-host /tmp path is invisible to it and the upload fails with
  "tar: empty archive".
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_UNIT = REPO_ROOT / "arena" / "deploy" / "chessarena-api.service"
ENGINE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-engine-build.yml"
ARENA_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-arena.yml"
DEPLOY_WRAPPER = REPO_ROOT / "arena" / "deploy" / "arena-deploy.sh"


def test_api_unit_starts_through_the_application_factory():
    assert API_UNIT.is_file(), f"missing {API_UNIT}"
    content = API_UNIT.read_text(encoding="utf-8")
    exec_line = next(
        line for line in content.splitlines() if line.startswith("ExecStart=")
    )

    assert "chessarena.main:create_app" in exec_line
    assert "--factory" in exec_line
    assert "chessarena.main:app" not in exec_line


def _scp_sources(content: str) -> list[str]:
    """Return the ``source:`` values of every appleboy/scp-action step."""
    sources = []
    # Iterate step blocks; find lines 'source: <value>' inside a step whose
    # 'uses:' is appleboy/scp-action.  The value may contain spaces (e.g.
    # GitHub expressions like "${{ steps.x.outputs.y }}"), so capture the
    # rest of the line rather than a whitespace-free token.
    blocks = re.split(r"\n\s*-\s+name:", content)
    for block in blocks:
        if "appleboy/scp-action" not in block:
            continue
        for line in block.splitlines():
            m = re.match(r"\s*source:\s*(.*?)\s*$", line)
            if m and m.group(1):
                sources.append(m.group(1).strip())
    return sources


def test_deploy_workflows_stage_archives_in_the_workspace():
    assert ENGINE_WORKFLOW.is_file(), f"missing {ENGINE_WORKFLOW}"
    assert ARENA_WORKFLOW.is_file(), f"missing {ARENA_WORKFLOW}"
    engine = ENGINE_WORKFLOW.read_text(encoding="utf-8")
    arena = ARENA_WORKFLOW.read_text(encoding="utf-8")

    # No scp-action source may point at a runner-host /tmp path (or any
    # absolute path): the Docker action cannot see the runner host /tmp.
    for src in _scp_sources(engine) + _scp_sources(arena):
        assert not src.startswith("/"), f"scp source must be relative: {src!r}"
        assert "/tmp" not in src, f"scp source must not use /tmp: {src!r}"

    # Engine workflow: the archive created in the staging step must be the
    # exact filename referenced by its upload step.
    archive_m = re.search(
        r'ARCHIVE="(build-[^\"]+\.tar\.gz)"', engine
    )
    assert archive_m, "engine workflow must define ARCHIVE=build-<id>.tar.gz"
    archive = archive_m.group(1)
    assert archive in _scp_sources(engine), (
        "engine upload source must equal the staged archive filename"
    )
    # Workspace-relative staging: tar must write "$ARCHIVE", never /tmp/...
    assert re.search(r"tar -C /tmp/build-stage -czf \"\$ARCHIVE\" \.", engine)
    # Archive-validity checks before upload.
    assert "test -s \"$ARCHIVE\"" in engine
    assert "tar -tzf \"$ARCHIVE\" >/dev/null" in engine

    # Arena workflow: packaged as arena.tar.gz and uploaded as arena.tar.gz.
    assert re.search(r"-C arena -czf arena\.tar\.gz \.", arena)
    assert "arena.tar.gz" in _scp_sources(arena)
    assert "test -s arena.tar.gz" in arena
    assert "tar -tzf arena.tar.gz >/dev/null" in arena


def test_clean_venv_test_step_runs_fixture_subprocesses_from_venv():
    """Executable fixtures use #!/usr/bin/env python3, so the clean-venv test
    step must put the venv bin on PATH before invoking pytest.  Otherwise the
    fake cutechess subprocess resolves /usr/bin/env python3 to the outer
    setup-python interpreter, which does not own the installed 'chess' module,
    and every fake-based test fails at setup (ModuleNotFoundError)."""
    content = ARENA_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"- name: Run arena tests from the clean venv\n(.*?)(?=\n\s*- name:|\n\s*- uses:|\Z)",
        content,
        re.DOTALL,
    )
    assert match, "missing 'Run arena tests from the clean venv' step"
    step = match.group(1)

    path_line = 'export PATH="/tmp/arena-venv/bin:$PATH"'
    pytest_line = "/tmp/arena-venv/bin/python -m pytest tests/ -q"

    assert path_line in step, "clean venv step must prepend the venv bin to PATH"
    assert pytest_line in step, "clean venv step must invoke the venv python"
    assert step.index(path_line) < step.index(pytest_line), (
        "PATH export must precede the pytest invocation"
    )


def test_deploy_wrapper_normalizes_working_directories():
    """pip/python/alembic inherit sys.path[0]='' resolved against the caller
    cwd.  The wrapper is invoked by the deploy user through sudo, whose SSH
    session starts in /home/deploy (750, not accessible to chessarena), so a
    bare 'sudo -u chessarena pip' crashes with PermissionError during pip's
    initial distribution scan.  The wrapper must therefore normalize its cwd
    to /opt/chessarena up front and run pip/Alembic from inside the release
    directory (alembic.ini's script_location is relative to the cwd)."""
    assert DEPLOY_WRAPPER.is_file(), f"missing {DEPLOY_WRAPPER}"
    content = DEPLOY_WRAPPER.read_text(encoding="utf-8")
    lines = content.splitlines()

    def line_index(needle: str) -> int:
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("#"):
                continue  # ignore comments that mention commands
            if needle in ln:
                return i
        return -1

    first_sudo = line_index("sudo -u chessarena")
    cd_root = line_index("cd /opt/chessarena")
    assert cd_root != -1, "wrapper must cd to /opt/chessarena"
    assert cd_root < first_sudo, (
        "wrapper must enter /opt/chessarena before any sudo -u chessarena"
    )

    cd_dest = line_index('cd "$dest"')
    assert cd_dest != -1, "release-install must enter the release directory"
    pip_index = line_index('pip" install -e .')
    assert pip_index != -1, "pip must run editable-install from the release dir"
    assert cd_dest < pip_index, "cd \"$dest\" must precede pip install -e ."

    alembic_index = line_index(
        '"$VENV/bin/alembic" -c alembic.ini upgrade head'
    )
    assert alembic_index != -1, "alembic must use the release-local alembic.ini"
    assert cd_dest < alembic_index, "alembic must run after cd \"$dest\""

    # Neither pip nor Alembic may bypass the normalized cwd via an absolute
    # target that would let an arbitrary caller cwd leak in.
    assert '"$VENV/bin/pip" install -e "$dest"' not in content
    assert '"$dest/alembic.ini"' not in content
