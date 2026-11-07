from pathlib import Path
import numpy as np
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent
WINDOW = 60

model = load_model(BASE / "btc_gru_returns.keras")
scaler = joblib.load(BASE / "r_scaler.pkl")
PAGE = (BASE / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="BTC Next-Day Predictor")


class PriceInput(BaseModel):
    prices: list[float]  # last 61 closing prices, oldest first


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def page():
    return PAGE


@app.post("/predict")
def predict(data: PriceInput):
    if len(data.prices) != WINDOW + 1:
        raise HTTPException(400, f"Send exactly {WINDOW + 1} prices")
    p = np.array(data.prices)
    if np.any(p <= 0):
        raise HTTPException(400, "Prices must be positive")

    rets = np.log(p[1:] / p[:-1]).reshape(-1, 1)
    x = scaler.transform(rets).reshape(1, WINDOW, 1)
    pred_ret = float(scaler.inverse_transform(model.predict(x, verbose=0))[0, 0])

    return {
        "last_price": float(p[-1]),
        "predicted_return_pct": round(pred_ret * 100, 3),
        "predicted_next_price": round(float(p[-1] * np.exp(pred_ret)), 2),
        "note": "Educational demo, not financial advice",
    }


@app.get("/live")
def live():
    resp = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": 90, "interval": "daily"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(503, "Price source unavailable, try again in a minute")
    prices = [p[1] for p in resp.json()["prices"]][-61:]
    return predict(PriceInput(prices=prices))
