
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import json

# ============================================================
# Agrishield Hedging Dashboard (UI + Payload Builder)
# - Streamlit collects inputs
# - Builds a single JSON payload for teammates to consume
# - Optional: POST payload to a backend endpoint (FastAPI/etc.)
# ============================================================

st.set_page_config(
    page_title="Agrishield Hedging Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COMMON_CROPS = [
    "Corn", "Soybeans", "Wheat", "Rice", "Cotton", "Barley", "Sorghum",
    "Oats", "Canola", "Sunflower", "Sugarcane", "Sugar Beet", "Potatoes",
    "Tomatoes", "Grapes", "Apples", "Alfalfa"
]

# -----------------------
# Helpers
# -----------------------
def normalize_crop(s: str) -> str:
    return " ".join(str(s).strip().split()).title()

def uniq_ci(items):
    seen = set()
    out = []
    for x in items:
        k = str(x).lower()
        if k not in seen:
            seen.add(k)
            out.append(str(x))
    return out

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- state init (BEFORE widgets) ----------
st.session_state.setdefault("quick_crops", [])
st.session_state.setdefault("manual_crops", [])
st.session_state.setdefault("crop_paste", "")
st.session_state.setdefault("crop_typed", "")
st.session_state.setdefault("manual_remove_selection", [])

# Store acreage as dict crop -> acres
st.session_state.setdefault("acreage_by_crop", {})

# Location: allow lat/lon explicitly (recommended for downstream)
st.session_state.setdefault("location", "")
st.session_state.setdefault("lat", None)
st.session_state.setdefault("lon", None)

# Back-end handoff option
st.session_state.setdefault("backend_url", "")
st.session_state.setdefault("backend_response", None)

def selected_crops_union():
    combined = [normalize_crop(c) for c in (st.session_state.quick_crops + st.session_state.manual_crops)]
    combined = [c for c in combined if c]
    return uniq_ci(combined)

def purge_manual_overlaps():
    """
    Enforce rule: a crop can live in only one bucket.
    If it's quick-selected, it should not exist in manual.
    """
    quick_set = {normalize_crop(c).lower() for c in st.session_state.quick_crops}
    st.session_state.manual_crops = [
        normalize_crop(c) for c in st.session_state.manual_crops
        if normalize_crop(c) and normalize_crop(c).lower() not in quick_set
    ]
    st.session_state.manual_crops = uniq_ci(st.session_state.manual_crops)

def add_manual_callback():
    purge_manual_overlaps()

    pasted = st.session_state.crop_paste
    typed = st.session_state.crop_typed

    to_add = []
    if str(pasted).strip():
        to_add += [ln for ln in str(pasted).splitlines() if ln.strip()]
    if str(typed).strip():
        to_add.append(str(typed))

    quick_set = {normalize_crop(c).lower() for c in st.session_state.quick_crops}

    manual = [normalize_crop(c) for c in st.session_state.manual_crops]
    manual = [c for c in manual if c]
    manual_set = {c.lower() for c in manual}

    for c in to_add:
        c2 = normalize_crop(c)
        if not c2:
            continue
        k = c2.lower()
        if k in quick_set:
            continue
        if k not in manual_set:
            manual.append(c2)
            manual_set.add(k)

    st.session_state.manual_crops = manual
    st.session_state.crop_paste = ""
    st.session_state.crop_typed = ""

def remove_manual_callback():
    rm = {normalize_crop(c).lower() for c in st.session_state.get("manual_remove_selection", [])}
    st.session_state.manual_crops = [
        c for c in st.session_state.manual_crops
        if normalize_crop(c).lower() not in rm
    ]
    st.session_state["manual_remove_selection"] = []

def sync_acreage_dict(selected_crops):
    """Keep acreage_by_crop aligned with selected crops."""
    acres = st.session_state.acreage_by_crop
    selected_set = set(selected_crops)

    # remove deleted crops
    for c in list(acres.keys()):
        if c not in selected_set:
            acres.pop(c, None)

    # add new crops with default 0
    for c in selected_crops:
        acres.setdefault(c, 0.0)

    st.session_state.acreage_by_crop = acres

def build_payload_from_state(ss: dict) -> dict:
    """
    Build a single, normalized payload that teammates can feed into:
    - climate pulls
    - soil pulls
    - risk/hedge optimization
    - LLM explanation
    """
    # location
    location_raw = (ss.get("location") or "").strip()
    lat = ss.get("lat", None)
    lon = ss.get("lon", None)

    # normalize lat/lon if possible
    lat_out = float(lat) if is_number(lat) else None
    lon_out = float(lon) if is_number(lon) else None

    # crops + acres
    selected = ss.get("selected_crops") or []
    acreage = ss.get("acreage_by_crop") or {}
    crops_payload = []
    for c in selected:
        name = normalize_crop(c)
        acres_val = acreage.get(c, acreage.get(name, 0.0))
        try:
            acres_val = float(acres_val)
        except Exception:
            acres_val = 0.0
        if acres_val < 0:
            acres_val = 0.0
        crops_payload.append({"name": name, "acres": acres_val})

    # planting window
    plant_start = ss.get("plant_start", None)
    plant_end = ss.get("plant_end", None)

    start_iso = plant_start.isoformat() if isinstance(plant_start, date) else None
    end_iso = plant_end.isoformat() if isinstance(plant_end, date) else None

    # preferences
    risk_aversion = ss.get("risk_aversion", 60.0)
    hedge_budget_pct = ss.get("hedge_budget_pct", 3.0)
    try:
        risk_aversion = float(risk_aversion)
    except Exception:
        risk_aversion = 60.0
    try:
        hedge_budget_pct = float(hedge_budget_pct)
    except Exception:
        hedge_budget_pct = 3.0
    risk_aversion = clamp(risk_aversion, 0.0, 100.0)
    hedge_budget_pct = clamp(hedge_budget_pct, 0.0, 100.0)

    payload = {
        "farm": {
            "location_raw": location_raw,
            "lat": lat_out,
            "lon": lon_out,
        },
        "crops": crops_payload,
        "planting_window": {
            "start": start_iso,
            "end": end_iso,
        },
        "preferences": {
            "risk_aversion_0_100": risk_aversion,
            "hedge_budget_pct_revenue": hedge_budget_pct,
        },
        "meta": {
            "units": {"acres": "acres"},
            "version": "1.0",
        },
    }
    return payload

def validate_payload(payload: dict) -> list[str]:
    errs = []

    # lat/lon required for downstream API pulls
    lat = payload.get("farm", {}).get("lat", None)
    lon = payload.get("farm", {}).get("lon", None)
    if lat is None or lon is None:
        errs.append("Latitude and longitude are required (enter them under Location).")
    else:
        if not (-90 <= float(lat) <= 90):
            errs.append("Latitude must be between -90 and 90.")
        if not (-180 <= float(lon) <= 180):
            errs.append("Longitude must be between -180 and 180.")

    crops = payload.get("crops", [])
    if not crops:
        errs.append("Select/add at least one crop.")
    else:
        if all((c.get("acres", 0.0) or 0.0) <= 0.0 for c in crops):
            errs.append("Enter acreage > 0 for at least one crop.")

    pw = payload.get("planting_window", {})
    if not pw.get("start") or not pw.get("end"):
        errs.append("Planting window start and end dates are required.")
    else:
        # ensure start <= end
        try:
            s = date.fromisoformat(pw["start"])
            e = date.fromisoformat(pw["end"])
            if e < s:
                errs.append("Planting window end date must be after start date.")
        except Exception:
            errs.append("Planting window dates are invalid.")

    return errs

# -----------------------
# SIDEBAR (ordered inputs)
# -----------------------
with st.sidebar:
    st.header("Farm inputs")

    # 1) Location
    st.subheader("Location")
    st.text_input(
        "Farm location (optional label)",
        key="location",
        placeholder="e.g., Ames, IA or 50010 (label only if lat/lon provided)"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Latitude",
            key="lat",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.lat if isinstance(st.session_state.lat, (int, float)) else 0.0,
            format="%.6f",
            help="Recommended: provide lat/lon so climate + soil pulls work reliably."
        )
    with col2:
        st.number_input(
            "Longitude",
            key="lon",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.lon if isinstance(st.session_state.lon, (int, float)) else 0.0,
            format="%.6f",
            help="Recommended: provide lat/lon so climate + soil pulls work reliably."
        )

    st.divider()

    # 2) Crops
    st.subheader("Crops")
    st.multiselect(
        "Quick pick (searchable) — unselect to remove",
        options=COMMON_CROPS,
        key="quick_crops",
        on_change=purge_manual_overlaps,
    )
    st.text_area("Paste crops (one per line)", key="crop_paste")
    st.text_input("Or type one crop", key="crop_typed")
    st.button("Add manual crops", on_click=add_manual_callback)

    selected_crops = selected_crops_union()
    st.session_state["selected_crops"] = selected_crops

    st.markdown("**Selected crops:**")
    if selected_crops:
        st.caption(", ".join(selected_crops))
    else:
        st.info("No crops added yet.")

    # Manual crop removal (quick crops removed by unselecting)
    if st.session_state.manual_crops:
        st.multiselect(
            "Remove manual crops",
            options=st.session_state.manual_crops,
            default=[],
            key="manual_remove_selection",
        )
        st.button("Remove selected manual crops", on_click=remove_manual_callback)

    # Recompute after potential removal so acreage updates cleanly
    selected_crops = selected_crops_union()
    st.session_state["selected_crops"] = selected_crops

    st.divider()

    # 3) Acreage (table)
    st.subheader("Acreage")
    if not selected_crops:
        st.caption("Add crops to enter acreage.")
    else:
        sync_acreage_dict(selected_crops)

        df = pd.DataFrame(
            [{"Crop": c, "Acres": float(st.session_state.acreage_by_crop.get(c, 0.0))}
             for c in selected_crops]
        )

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            disabled=["Crop"],
            column_config={
                "Acres": st.column_config.NumberColumn(
                    "Acres",
                    min_value=0.0,
                    step=1.0,
                    format="%.1f",
                    help="Enter acreage for each crop."
                )
            },
            key="acreage_table",
        )

        st.session_state.acreage_by_crop = {
            row["Crop"]: float(row["Acres"]) for _, row in edited.iterrows()
        }

        total_acres = sum(st.session_state.acreage_by_crop.values())
        st.caption(f"Total acreage: {total_acres:,.1f}")

    st.divider()

    # 4) Planting window
    st.subheader("Planting window")
    default_start = date.today()
    default_end = date.today() + timedelta(days=30)
    planting_window = st.date_input("Start / end dates", value=(default_start, default_end))
    if isinstance(planting_window, tuple) and len(planting_window) == 2:
        st.session_state["plant_start"], st.session_state["plant_end"] = planting_window
    else:
        st.session_state["plant_start"] = planting_window
        st.session_state["plant_end"] = planting_window

    st.divider()

    # 5) Risk aversion
    st.subheader("Risk tolerance")
    st.session_state["risk_aversion"] = st.slider(
        "Risk aversion",
        min_value=0,
        max_value=100,
        value=int(st.session_state.get("risk_aversion", 60)),
        help="0 = risk seeking, 100 = very risk averse",
    )

    st.divider()

    # 6) Hedge budget (number input)
    st.subheader("Hedge budget")
    st.session_state["hedge_budget_pct"] = st.number_input(
        "Percent of revenue allocated to hedging (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("hedge_budget_pct", 3.0)),
        step=0.25,
        format="%.2f",
        help="Example: 3.00 means 3% of revenue."
    )

    st.divider()

    # Optional: backend endpoint for teammates (FastAPI/etc.)
    st.subheader("Backend integration (optional)")
    st.text_input(
        "Backend URL (POST /analyze)",
        key="backend_url",
        placeholder="http://localhost:8000/analyze"
    )

# -----------------------
# MAIN PAGE
# -----------------------
st.title("Agrishield Hedging Dashboard")
st.caption("UI collects farm inputs and exports a standardized JSON payload for climate/soil/hedging pipelines.")

payload = build_payload_from_state(st.session_state)
errors = validate_payload(payload)

# Show validation status
colA, colB = st.columns([2, 1])
with colA:
    st.subheader("Input status")
    if errors:
        st.error("Fix the issues below to generate a valid payload.")
        for e in errors:
            st.write(f"- {e}")
    else:
        st.success("Inputs look good. Payload is ready.")

with colB:
    st.subheader("Quick summary")
    total_acres = sum((c.get("acres", 0.0) or 0.0) for c in payload["crops"])
    st.metric("Crops", len(payload["crops"]))
    st.metric("Total acres", f"{total_acres:,.1f}")
    st.metric("Risk aversion", f'{payload["preferences"]["risk_aversion_0_100"]:.0f}/100')

st.divider()

st.subheader("Handoff payload")
st.write("This JSON is the contract your teammates should code against (climate/soil pulls, loss function, optimization, LLM explanation).")
st.json(payload, expanded=False)

# Save payload in session_state for reuse elsewhere
st.session_state["payload"] = payload

# Download button
payload_text = json.dumps(payload, indent=2)
st.download_button(
    "Download payload.json",
    data=payload_text,
    file_name="payload.json",
    mime="application/json"
)

# Optional: POST to backend
if st.session_state.get("backend_url"):
    st.subheader("Send to backend")
    st.write("If your teammates run a backend service, you can POST the payload and display the response here.")
    send = st.button("POST payload to backend")
    if send:
        if errors:
            st.error("Cannot send payload: fix input errors first.")
        else:
            try:
                import requests
                resp = requests.post(st.session_state["backend_url"], json=payload, timeout=30)
                st.session_state["backend_response"] = {
                    "status_code": resp.status_code,
                    "json": resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text
                }
            except Exception as ex:
                st.session_state["backend_response"] = {"error": str(ex)}

    if st.session_state.get("backend_response") is not None:
        st.markdown("**Backend response:**")
        st.json(st.session_state["backend_response"], expanded=True)

st.divider()

# Placeholder output area (what your teammates will eventually fill)
st.subheader("Outputs (placeholder)")
st.info(
    "Next: plug in your teammates' pipeline outputs here (risk scores, hedge ratios, explanations). "
    "For now, this page focuses on collecting and exporting clean inputs."
)

