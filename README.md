# 🌾 AgriShield

> **AI-powered climate risk quantification and hedging for farmers**  
> Built for the IBM Watsonx Hackathon · Powered by IBM Granite

---

## The Problem

Agriculture is one of the most financially vulnerable industries to climate volatility. A single bad season — a drought, a heat wave, an anomalous ENSO cycle — can wipe out a farmer's entire year of revenue. Yet small and mid-sized farmers lack accessible tools to quantify that risk or protect against it. Existing crop insurance is slow, opaque, and rarely personalized to a specific farm's real-time conditions.

**AgriShield fixes this.** It transforms raw climate and soil data into a personalized, actionable financial protection strategy — in seconds.

---

## How It Works

AgriShield is a three-layer climate-risk engine:

```
Soil + Weather Data
        │
        ▼
┌───────────────────┐
│  1. Agri Data     │  Soil quality score, current anomaly signal
│     Module        │  (how hot/dry this season already looks)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. Seasonal      │  ENSO phase + 14-day forecast →
│     Outlook       │  probabilistic view of the coming season
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. Monte Carlo   │  10,000 simulations over 30+ years of NASA
│     Engine        │  weather data, weighted by climate change,
│                   │  ENSO phase, and current anomaly →
│                   │  full yield-loss distribution per crop
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  IBM Granite LLM  │  Converts quantitative data into a structured
│  (Watsonx)        │  loss model + plain-language risk explanation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Hedge Optimizer  │  Recommends practical commodity options
│                   │  strategies to protect expected revenue
└───────────────────┘
```

---

## Modules

| File | Description |
|------|-------------|
| `1_agri_data.py` | Pulls soil properties and recent weather for a farm location |
| `2_seasonal_outlook.py` | Reads ENSO state and 14-day forecast for regional risk outlook |
| `3_monte_carlo.py` | Simulates thousands of growing seasons for yield-loss distributions |
| `4_market_data.py` | Fetches commodity prices and calculates full revenue baselines |
| `5_yield_risk.py` | Granite LLM Call 1 — structured JSON risk assessment per crop |

---

## Example Output

```
Farm: NW Arkansas | Crops: Rice (50ac), Wheat (20ac), Soybeans (30ac)
ENSO: Weak La Niña | Signal: DRY (-54.9% precip anomaly)

Rice      │ Risk: 72/100 │ P50 loss: $25,161 (39%) │ Driver: Drought × low soil buffer
Wheat     │ Risk: 68/100 │ P50 loss:  $2,328 (40%) │ Driver: Precip deficit compounding
Soybeans  │ Risk: 61/100 │ P50 loss:  $5,299 (29%) │ Driver: Fat bad tail (skew -0.69)

Total Revenue at Risk:  $88,376
Median Season Loss:     $32,788  (37.1%)
Worst-Case Loss (P10):  $60,006  (67.9%)
```

---

## Tech Stack

- **IBM Granite** (`ibm/granite-4-0-tiny-preview`) via IBM Watsonx
- **NASA / NOAA** climate data for 30-year historical baselines
- **ENSO / ONI** indices for seasonal outlook modeling
- **Monte Carlo simulation** — 10,000 runs per crop with climate-weighted sampling
- **Black-Scholes options pricing** for hedge recommendation
- Python · `requests` · `numpy` · `scipy`

---

## Setup

```bash
git clone https://github.com/ishanpatel28/ArgiShield.git
cd ArgiShield
pip install -r requirements.txt
```

Create a `.env` file with your IBM Watsonx credentials:
```
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
```

Run the full pipeline:
```bash
python 1_agri_data.py
python 2_seasonal_outlook.py
python 3_monte_carlo.py
python 4_market_data.py
python 5_yield_risk.py
```

---

## Why IBM Granite?

Granite sits at the critical junction between raw quantitative data and actionable farmer guidance. The Monte Carlo engine produces distributions — Granite interprets them. It reasons about how ENSO history interacts with soil pH, how distribution skewness signals hidden tail risk, and what compounding drought + heat stress means agronomically for each specific crop. This is context that math alone cannot capture.

---

*Built for the IBM Watsonx Hackathon 2026*
