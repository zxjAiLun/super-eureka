"""Opening set registry API (section 16.3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import OpeningSet
from ..schemas import OpeningSetOut

router = APIRouter(tags=["openings"])


@router.get("/opening-sets", response_model=list[OpeningSetOut])
def list_opening_sets(session: Session = Depends(get_db)):
    return (
        session.query(OpeningSet)
        .order_by(OpeningSet.created_at.desc())
        .all()
    )


@router.get("/opening-sets/{opening_set_id}", response_model=OpeningSetOut)
def get_opening_set(opening_set_id: str, session: Session = Depends(get_db)):
    opening = (
        session.query(OpeningSet)
        .filter(OpeningSet.opening_set_id == opening_set_id)
        .first()
    )
    if opening is None:
        raise HTTPException(status_code=404, detail="opening set not found")
    return opening
