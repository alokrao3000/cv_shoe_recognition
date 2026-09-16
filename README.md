# Sneaker Identification Engine

Given a retailer product page URL and/or sneaker photo(s), identify the **exact
release** — brand, model, colorway, **style code** — and map it to the
**verified StockX product** with its lowest ask / highest bid. When the evidence
is weak, say so and queue it for human review instead of guessing.

```
URL / photos ─► extract page ─► detect style codes ─► vision analysis (Claude)
             ─► visual similarity (DINOv2 + pgvector) ─► web + StockX candidate search
             ─► score every candidate (transparent weights) ─► contradiction checks
             ─► verifier pass (Claude) ─► StockX confirmation ─► tier: HIGH / MEDIUM / LOW / UNRESOLVED
```

Companion to `sneaker-arbitrage`: that scraper drops products it can't parse a
SKU from (`sku_parse_failed`); this service resolves them. StockX is the
**destination** (official API), never scraped.

## Quick start

```
python -m venv venv && venv\Scripts\pip install -r requirements.txt
copy .env.example .env                    # fill in keys (see below)
docker compose up -d postgres             # Postgres 16 + pgvector on :5434
venv\Scripts\python -m uvicorn app.main:app --reload --port 8100
```

Open http://localhost:8100 — drop photos and/or paste a product URL, watch the
stages, get the result. `/review` is the human-verification queue, `/docs` the API.

### Keys

| Setting | Needed for | Without it |
|---|---|---|
| `STOCKX_*` + `python scripts/stockx_auth.py` | candidate discovery by name/code, product confirmation, market data | no StockX mapping, `STOCKX_API_FAILED` |
| `ANTHROPIC_API_KEY` | vision analysis (brand/model/colorway/tag text) and the verifier | `VISION_UNAVAILABLE`; HIGH only reachable via a page-printed SKU confirmed by StockX |
| `SERPER_API_KEY` | web discovery of style codes; reference photos for image-less candidates | `WEB_SEARCH_FAILED`; relies on page SKU + StockX + local index |

Every signal is optional; the pipeline records what was unavailable in the
evidence and never fabricates confidence to compensate.

### Seed the reference catalog

```
python scripts/seed_from_arbitrage_db.py            # products + photos from sneaker-arbitrage's DB (needs its postgres up)
python scripts/seed_stockx_catalog.py --images 300  # widen with StockX catalog + one web photo per product
```

Both are incremental and safe to re-run. The index also grows on its own:
reference photos fetched during identifications and every human-confirmed
match are stored as labeled `product_images`.

## How identification works

**Inputs** — up to 6 images (upload, `image_url`) and/or a `product_url`, `title`,
`description`. Identical inputs hit the resolution cache (`identifications.cache_key`).

**Stages** (`app/pipeline/identify.py`; each writes to the evidence graph and is timed):

1. `extract` — `app/extraction/page_extractor.py`: JSON-LD/microdata → Shopify
   `/products/<handle>.json` → `__NEXT_DATA__`/embedded JSON → OpenGraph/meta →
   HTML. Collects title, brand, price, breadcrumbs, identifiers, all product images.
   SSRF-guarded (http(s), public IPs, size/time bounded).
2. `sku` — `app/extraction/sku_detector.py`: brand-aware style-code formats
   (Nike/Jordan, adidas, New Balance, ASICS, Vans, Converse, Puma, Salomon, …),
   labeled > bare, dates/prices/UPCs rejected, `DD1391-100 ≡ DD1391100 ≡ DD1391 100`.
3. `vision` — `app/vision/claude_vision.py`: one structured call over all images
   (brand, model incl. height, colorway in official order, nickname, legible tag
   code, gender/GS cues, distinctive features, likely releases). Model:
   `VISION_MODEL` (default `claude-opus-5`).
4. `embedding` — DINOv2 CLS embeddings (`app/vision/embeddings.py`, kept from the
   original project: CLS beat patch pooling 57.5% vs 22.5% top-1 in a real
   cross-retailer eval) searched with pgvector cosine (`app/db/vector_search.py`).
5. `candidates` — `app/search/web_discovery.py` (Serper) harvests style codes from
   result titles/snippets, weighting agreement across domains and domain trust;
   StockX `catalog/search` by title / vision attributes; every code resolved to
   StockX metadata (`app/stockx/catalog.py`, DB-cached); image-less leaders get a
   web reference photo embedded for the visual signal.
6. `resolve` — `app/pipeline/scoring.py`:

   | component | weight | what it compares |
   |---|---|---|
   | sku_match | 0.40 | page-printed code 1.0 · tag code 0.9 · web agreement 0.35+0.15/domain · vision recall 0.35 |
   | model_match | 0.20 | canonical model (`Nike Dunk Low` ≠ `Nike Dunk High`) |
   | colorway_match | 0.15 | ordered colour tokens (`White/Black` ≠ `Black/White`), nickname equality |
   | embedding_similarity | 0.15 | cosine vs reference photos, floor 0.70 → ceiling 0.92 |
   | metadata_match | 0.05 | price vs retail, gender, size category, brand |
   | external_agreement | 0.05 | number of independent signals |

   Unknown = neutral (0.5); missing ≠ contradicting. **Contradictions** then
   reject (model mismatch, GS/PS/TD vs adult, brand mismatch, non-footwear) or
   penalize (gender, colorway conflict, tag/page code mismatch, missing collab,
   inconsistent input images). Weights/thresholds are all `.env` settings.
7. `verify` — the top 3 survivors go back to Claude with the query photos (and
   reference photos when available) as a *skeptical* verifier: same / different /
   uncertain with concrete contradictions. `different` rejects, `uncertain` caps
   below HIGH.
8. `stockx` — winner's style code must resolve to exactly one StockX product whose
   title agrees; otherwise `STOCKX_MATCH_UNCERTAIN`. `market` fetches per-size
   asks/bids (withheld when unresolved).

**Tiers** — ≥0.95 HIGH, ≥0.85 MEDIUM, ≥0.70 LOW, else UNRESOLVED. HIGH additionally
requires verifier agreement *or* a page-printed style code confirmed by StockX.
Top-2 within `AMBIGUITY_MARGIN` → `MULTIPLE_CANDIDATES` and never HIGH. LOW,
UNRESOLVED, `MULTIPLE_CANDIDATES` and `STOCKX_MATCH_UNCERTAIN` go to `/review`.

**Failure codes**: `NO_PRODUCT_IMAGE NO_METADATA NO_SKU VISION_UNCERTAIN
VISION_UNAVAILABLE MULTIPLE_CANDIDATES WEB_SEARCH_FAILED STOCKX_API_FAILED
STOCKX_MATCH_UNCERTAIN CONTRADICTORY_EVIDENCE NOT_A_SNEAKER NO_CANDIDATES`.

## API

```
POST /api/identify              multipart: files[], product_url, image_url, title, description → {id}
GET  /api/identify/{id}         stages, status, confidence, product, stockx, candidates, evidence, failure_codes
POST /api/identify/{id}/rerun   fresh run from the same inputs
GET  /api/identifications       recent
POST /api/sneaker/identify      JSON {image_url|image_urls, product_url, title, description} → synchronous result
GET  /api/review/queue          items needing a human
POST /api/review/{id}/decision  {action: confirm|reject|select|manual_sku, style_code?, notes?}
GET  /health
```

Set `API_TOKEN` to require a bearer token on `/api/*`; requests are rate-limited
per IP; uploads are size/type checked; fetched URLs must be public http(s).

The arbitrage engine should treat the response as:
`high` → use `stockx` automatically · `medium` → use but mark for verification ·
`low` → don't use for profit calculations · `unresolved` → manual review.

## Evidence graph

Every identification stores `evidence` (input, page, detected attributes, vision,
visual/web/StockX candidates, scoring ranking, contradictions, verification,
resolution, errors, timings) plus every candidate's per-component scores and
notes. If it's wrong, the UI's Evidence panel shows *which* signal misled it.

## Human review → training data

`/review` shows the retailer product beside the scored candidates. Confirm /
select / enter a style code / reject. A human match sets the identification to
HIGH, upserts the product with StockX metadata, and stores the query photos as
`user_confirmed` reference images — so the next photo of that shoe matches
visually. Decisions are kept in `review_decisions` with what the system had said.

## Evaluation

```
python scripts/build_eval_set.py --pairs 100     # cross-retailer held-out pairs from the arbitrage DB (+ data/eval/curated.jsonl)
python scripts/run_eval.py                       # top-1/top-3, SKU acc, StockX mapping acc, FALSE-POSITIVE rate, unresolved, latency
python scripts/run_eval.py --no-llm              # ablations: --no-web --no-db --no-url
```

The false-positive rate (confident but wrong) is the headline metric — a wrong
SKU silently feeding the arbitrage engine is the failure this project exists to
prevent. Add hard cases (similar colorways, GS vs adult, collabs, poor photos) to
`data/eval/curated.jsonl`; tune weights/thresholds in `.env` from the results.

Unit tests (`venv\Scripts\pytest`) cover extraction on fixture pages, style-code
detection, normalization, web discovery, scoring/contradictions/tiers on the
spec's scenarios, and a mocked end-to-end pipeline (no network, no DB, no torch).

## Layout

```
app/extraction/   page_extractor · sku_detector · normalize
app/vision/       embeddings (DINOv2) · claude_vision
app/search/       provider (Serper, swappable) · web_discovery
app/stockx/       client (official API, OAuth, budget) · catalog (cached)
app/pipeline/     identify (orchestrator) · candidates · scoring · service (persistence) · cache
app/db/           models (pgvector) · session · vector_search · repo
app/review/       review API
app/static/       UI (vanilla JS)
scripts/          stockx_auth · seed_from_arbitrage_db · seed_stockx_catalog · build_eval_set · run_eval · identify_cli
```

## Known limitations

- Reference coverage is what has been seeded/confirmed; a never-seen shoe relies
  on the page SKU, vision + web discovery and StockX — which is the designed path.
- No fine-tuning: DINOv2 is used zero-shot; the visual signal is one of six and
  can't produce a confident answer alone by construction.
- StockX's API has no images; reference photos come from retailers, the web and
  confirmed uploads.
- Not yet wired into sneaker-arbitrage's `sku_parse_failed` path — call
  `POST /api/sneaker/identify` from there once the eval numbers are acceptable.
