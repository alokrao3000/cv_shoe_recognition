from typing import List

from fastapi import APIRouter, HTTPException, Request

from app.schemas import IdentifyResult, ReviewDecisionRequest, ReviewQueueItem

router = APIRouter(prefix="/api/review", tags=["review"])


def _service(request: Request):
    return request.app.state.service


@router.get("/queue", response_model=List[ReviewQueueItem])
def review_queue(request: Request, limit: int = 50):
    return _service(request).queue(limit=min(limit, 200))


@router.get("/{ident_id}", response_model=IdentifyResult)
def review_item(ident_id: str, request: Request):
    result = _service(request).get(ident_id)
    if result is None:
        raise HTTPException(404, "identification not found")
    return result


@router.post("/{ident_id}/decision", response_model=IdentifyResult)
def review_decide(ident_id: str, decision: ReviewDecisionRequest, request: Request):
    try:
        result = _service(request).decide(ident_id, decision)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if result is None:
        raise HTTPException(404, "identification not found")
    return result
