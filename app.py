import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime

st.set_page_config(page_title="Prediksi Harga Crypto Tokocrypto", layout="wide")
st.title("📈 Prediksi Harga Crypto (Data Binance/Tokocrypto)")

@st.cache_data
def get_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 1000}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                     "_1", "_2", "_3", "_4", "_5", "_6"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["target"] = df["close"].shift(-1)
    return df.dropna()

df = get_data()

if len(df) > 10:
    X = df[["open", "high", "low", "volume"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.success(f"Model Akurasi: {score:.2f}")

    st.subheader("📊 Data Terbaru")
    st.dataframe(df.tail())

    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(title="Candlestick BTC/USDT (Binance)", xaxis_title="Waktu", yaxis_title="Harga (USDT)",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    last_data = X.tail(1)
    pred = model.predict(last_data)[0]
    st.success(f"🎯 Prediksi Harga Selanjutnya: ${pred:,.2f}")
else:
    st.warning("Data terlalu sedikit untuk pelatihan model.")
