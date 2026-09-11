import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app import embeddings, stockx_client
from app.config import settings
from app.index import ReferenceIndex, classify_match
from app.models import IdentifyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV Shoe Recognition")

_index: "ReferenceIndex | None" = None
_stockx = stockx_client.StockXAPIClient()


@app.on_event("startup")
def _startup():
    global _index
    try:
        _index = ReferenceIndex.load()
        logger.info(f"Loaded reference index with {len(_index)} SKUs.")
    except FileNotFoundError as exc:
        logger.warning(f"{exc} The /identify endpoint will 503 until it's built.")


@app.on_event("shutdown")
def _shutdown():
    _stockx.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "reference_index_loaded": _index is not None,
        "reference_index_size": len(_index) if _index is not None else 0,
        "stockx_configured": stockx_client.is_configured(),
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    if _index is None:
        raise HTTPException(503, "Reference index not built yet — run scripts/build_reference_index.py.")

    data = await file.read()
    try:
        image = embeddings.load_image(data)
    except Exception:
        raise HTTPException(400, "Couldn't read that as an image.")

    vec = embeddings.embed_image(image)
    candidates = _index.search(vec)
    identified, reason = classify_match(candidates)

    resp = IdentifyResponse(identified=identified, reason=reason, candidates=candidates,
                            top_match=candidates[0] if candidates else None)

    if identified:
        if not stockx_client.is_configured():
            resp.market_error = "stockx_not_configured"
        else:
            market, err = _stockx.get_market(resp.top_match.sku, resp.top_match.name or "")
            if market is not None:
                resp.market = market
            else:
                resp.market_error = err

    return resp


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <!doctype html>
    <html>
    <head><title>CV Shoe Recognition</title></head>
    <body style="font-family: system-ui; max-width: 480px; margin: 40px auto;">
      <h2>Identify a shoe photo</h2>
      <form action="/identify" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Identify</button>
      </form>
      <p>Or POST to <code>/identify</code> directly (multipart <code>file</code> field).
      See <a href="/docs">/docs</a> for the full API.</p>
    </body>
    </html>
    """
