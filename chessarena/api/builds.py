"""Engine build registry API (section 16.2).

v1 does NOT allow uploading binaries through the public API; builds are
installed out-of-band by the deploy pipeline and the ``install_build`` script,
which register the immutable directory in the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import EngineBuild
from ..schemas import BuildOut

router = APIRouter(tags=["builds"])


@router.get("/builds", response_model=list[BuildOut])
def list_builds(session: Session = Depends(get_db)):
    return (
        session.query(EngineBuild)
        .order_by(EngineBuild.created_at.desc())
        .all()
    )


@router.get("/builds/{build_id}", response_model=BuildOut)
def get_build(build_id: str, session: Session = Depends(get_db)):
    build = (
        session.query(EngineBuild)
        .filter(EngineBuild.build_id == build_id)
        .first()
    )
    if build is None:
        raise HTTPException(status_code=404, detail="build not found")
    return build
