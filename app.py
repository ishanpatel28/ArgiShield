"""
app.py
======
AgriShield – FastAPI Application
----------------------------------
Routes:
  GET  /           → form page (index.html)
  POST /analyze    → run full pipeline, return JSON
  GET  /results    → results page (reads from session via query param or stored state)
  POST /pdf        → generate and stream PDF report
"""

import json
import logging
import os
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("agrishield.app")

app = FastAPI(title="AgriShield", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Custom Jinja2 filters
def _format_thousands(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"

templates.env.filters["format_thousands"] = _format_thousands

# In-memory result store (keyed by session id)
# For production you'd use Redis or a DB — fine for a hackathon demo
_results_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_crops_form(crops_raw: str, acres_raw: str) -> list[dict]:
    """
    Parse crop names and acreages from the form.
    crops_raw: comma-separated crop names, e.g. "Rice, Wheat, Soybeans"
    acres_raw: comma-separated acreages,   e.g. "50, 20, 30"
    """
    names  = [n.strip().title() for n in crops_raw.split(",") if n.strip()]
    acres  = [a.strip() for a in acres_raw.split(",")]
    result = []
    for i, name in enumerate(names):
        try:
            ac = float(acres[i]) if i < len(acres) else 0.0
        except ValueError:
            ac = 0.0
        if name:
            result.append({"name": name, "acres": max(0.0, ac)})
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(
    request: Request,
    location: str = Form(""),
    lat:      str = Form(""),
    lon:      str = Form(""),
    crops:    str = Form(""),
    acres:    str = Form(""),
    plant_start: str = Form(""),
    plant_end:   str = Form(""),
    risk_aversion:    str = Form("60"),
    hedge_budget_pct: str = Form("3.0"),
):
    # Build farm_input
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except ValueError:
        return JSONResponse(
            {"error": "Invalid latitude or longitude. Please enter numeric coordinates."},
            status_code=400,
        )

    crop_list = _parse_crops_form(crops, acres)
    if not crop_list:
        return JSONResponse({"error": "Please enter at least one crop."}, status_code=400)
    if all(c["acres"] == 0 for c in crop_list):
        return JSONResponse({"error": "Please enter acreage > 0 for at least one crop."}, status_code=400)

    try:
        ra  = float(risk_aversion)
        hb  = float(hedge_budget_pct)
    except ValueError:
        ra, hb = 60.0, 3.0

    farm_input = {
        "farm": {
            "location_raw": location,
            "lat":  lat_f,
            "lon":  lon_f,
        },
        "crops": crop_list,
        "planting_window": {
            "start": plant_start,
            "end":   plant_end,
        },
        "preferences": {
            "risk_aversion_0_100":      max(0.0, min(100.0, ra)),
            "hedge_budget_pct_revenue": max(0.0, min(100.0, hb)),
        },
        "meta": {"version": "1.0"},
    }

    # Run pipeline
    log.info("Starting pipeline for lat=%.4f lon=%.4f crops=%s", lat_f, lon_f, [c["name"] for c in crop_list])
    try:
        from pipeline import run_pipeline
        result = run_pipeline(farm_input)
    except Exception as exc:
        log.exception("Pipeline crashed: %s", exc)
        return JSONResponse({"error": f"Pipeline error: {exc}"}, status_code=500)

    # Store result and redirect
    session_id = uuid.uuid4().hex
    _results_store[session_id] = result

    return JSONResponse({"session_id": session_id, "redirect": f"/results?id={session_id}"})


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request, id: Optional[str] = None):
    if not id or id not in _results_store:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "No results found. Please submit the form."},
        )
    result = _results_store[id]
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "result": result, "session_id": id},
    )


@app.post("/pdf/{session_id}")
async def generate_pdf(session_id: str):
    if session_id not in _results_store:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    result = _results_store[session_id]

    try:
        from pdf_report import build_pdf
        pdf_bytes = build_pdf(result)
    except Exception as exc:
        log.exception("PDF generation failed: %s", exc)
        return JSONResponse({"error": f"PDF error: {exc}"}, status_code=500)

    location = result.get("farm_input", {}).get("farm", {}).get("location_raw", "farm") or "farm"
    filename  = f"agrishield_{location.replace(' ', '_').lower()}.pdf"

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
