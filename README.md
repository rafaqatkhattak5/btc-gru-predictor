# Bitcoin Next-Day Predictor (GRU + FastAPI)

Screenshot_20261107-130325.jpg

A deep learning practice project: LSTM and GRU models for Bitcoin time series, deployed as a FastAPI web app with a mobile-friendly UI.

## What I built
- Data: daily BTC-USD prices from Yahoo Finance (2018 to 2026)
- Models: LSTM and GRU (TensorFlow/Keras), 60-day windows
- Leak-free pipeline: time-ordered split, scaler fit on train only, EarlyStopping
- Feature engineering: returns, volume change, moving-average ratios, RSI, volatility
- API: FastAPI with input validation (`/predict`, `/live`, `/health`)
- UI: single-page HTML app that fetches live prices from CoinGecko

## Results (test set, always compared to a naive baseline)

| Model | RMSE | Direction accuracy |
|---|---|---|
| Naive (tomorrow = today) | $1,934 | n/a |
| GRU on returns | $1,933 | 47.1% |
| GRU on 6 features | $1,946 | 49.4% |
| GRU on raw prices | $2,268 | n/a |
| LSTM on raw prices | $3,371 | n/a |

## Honest conclusion
None of the models beat the naive baseline. Bitcoin's next-day move is very close to a random walk, so past prices carry almost no signal. Raw-price models also suffer from scale shift: the test period reached prices never seen in training. The value of this project is the full pipeline and the baseline check, not the prediction.

## Run locally
    pip install -r requirements.txt
    uvicorn app:app --reload

Then open http://127.0.0.1:8000

## Disclaimer
Educational project only. Not financial advice.
