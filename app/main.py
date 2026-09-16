import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import db_available, init_db
from app.pipeline.identify import RunInput
from app.pipeline.service import IdentificationService
from app.review.routes import router as review_router
from app.schemas import IdentifyRequest, IdentifyResult, ReviewQueueItem
from app.search.provider import get_provider
from app.stockx import client as stockx_client
from app.vision.claude_vision import VisionClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(settings.data_dir) / "uploads"

app = FastAPI(title="Sneaker Identification Engine", version="1.0")
app.include_router(review_router)


# ── security middleware: optional bearer token + per-IP rate limit on /api/* ──

_buckets: dict = defaultdict(deque)
_bucket_lock = threading.Lock()


@app.middleware("http")
async def _guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/"):
        if settings.api_token:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {settings.api_token}":
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        ip = request.client.host if request.client else "?"
        now = time.monotonic()
        with _bucket_lock:
            q = _buckets[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= settings.rate_limit_per_minute:
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
            q.append(now)
    return await call_next(request)


@app.on_event("startup")
def _startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.state.db_ok = db_available()
    if app.state.db_ok:
        init_db()
    else:
        logger.warning(f"Database unreachable at {settings.database_url.split('@')[-1]} — "
                       "start it with `docker compose up -d postgres`. /api/identify will 503 until then.")
    app.state.service = IdentificationService()
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.on_event("shutdown")
def _shutdown():
    try:
        app.state.service.pipeline.catalog.close()
    except Exception:
        pass


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/review", include_in_schema=False)
def review_page():
    return FileResponse(STATIC_DIR / "review.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": db_available(),
        "stockx_configured": stockx_client.is_configured(),
        "vision_configured": VisionClient().available,
        "vision_model": settings.vision_model,
        "search_provider": get_provider().name if get_provider().available else None,
    }


def _require_db():
    if not db_available():
        raise HTTPException(503, "database unavailable — run `docker compose up -d postgres`")


async def _read_uploads(files: List[UploadFile]) -> List[bytes]:
    out = []
    for f in files[: settings.max_images_per_request]:
        if f.content_type and not (f.content_type.startswith("image/") or f.content_type == "application/octet-stream"):
            raise HTTPException(400, f"{f.filename}: not an image ({f.content_type})")
        data = await f.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"{f.filename}: larger than {settings.max_upload_bytes // (1024 * 1024)} MB")
        if data:
            out.append(data)
    return out


def _validate_url(url: Optional[str]) -> str:
    url = (url or "").strip()
    if url and not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "URLs must start with http:// or https://")
    return url


@app.post("/api/identify")
async def identify(background: BackgroundTasks,
                   files: List[UploadFile] = File(default=[]),
                   product_url: Optional[str] = Form(default=None),
                   image_url: Optional[str] = Form(default=None),
                   title: Optional[str] = Form(default=None),
                   description: Optional[str] = Form(default=None)):
    """Starts an identification. Poll GET /api/identify/{id} for progress and the result."""
    _require_db()
    images = await _read_uploads(files)
    inp = RunInput(images=images, image_urls=[u for u in [_validate_url(image_url)] if u],
                   product_url=_validate_url(product_url), title=(title or "").strip()[:500],
                   description=(description or "").strip()[:5000])
    if not (inp.images or inp.image_urls or inp.product_url or inp.title):
        raise HTTPException(400, "provide at least one image, an image URL, a product URL or a title")
    service: IdentificationService = app.state.service
    ident_id = service.create(inp)
    background.add_task(service.run, ident_id)
    return {"id": ident_id, "status": "queued"}


@app.get("/api/identify/{ident_id}", response_model=IdentifyResult)
def identify_status(ident_id: str):
    _require_db()
    result = app.state.service.get(ident_id)
    if result is None:
        raise HTTPException(404, "identification not found")
    return result


@app.post("/api/identify/{ident_id}/rerun")
def identify_rerun(ident_id: str, background: BackgroundTasks):
    _require_db()
    new_id = app.state.service.rerun(ident_id)
    if new_id is None:
        raise HTTPException(404, "identification not found")
    background.add_task(app.state.service.run, new_id)
    return {"id": new_id, "status": "queued"}


@app.get("/api/identifications", response_model=List[ReviewQueueItem])
def identifications(limit: int = 20):
    _require_db()
    return app.state.service.recent(limit=min(limit, 100))


@app.post("/api/sneaker/identify", response_model=IdentifyResult)
def identify_sync(req: IdentifyRequest):
    """Synchronous JSON endpoint for the arbitrage scraper (spec §24). Runs
    the full pipeline inline and returns the finished result."""
    _require_db()
    urls = [u for u in ([req.image_url] if req.image_url else []) + list(req.image_urls) if u]
    for u in urls + ([req.product_url] if req.product_url else []):
        _validate_url(u)
    inp = RunInput(image_urls=urls, product_url=req.product_url or "", title=(req.title or "")[:500],
                   description=(req.description or "")[:5000])
    if not (inp.image_urls or inp.product_url or inp.title):
        raise HTTPException(400, "provide image_url(s), product_url or title")
    service: IdentificationService = app.state.service
    ident_id = service.create(inp)
    service.run(ident_id)
    return service.get(ident_id)
