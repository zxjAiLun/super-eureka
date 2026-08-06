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
