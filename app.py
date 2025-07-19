import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi Harga Crypto Tokocrypto", layout="wide")
st.title("\U0001F4C8 Prediksi Harga Crypto (Data Binance/Tokocrypto)")

@st.cache_data
def get_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 1000}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "_1", "_2", "_3", "_4", "_5", "_6"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["target"] = df["close"].shift(-1)
    return df.dropna()

df = get_data()

# Model Training
X = df[["open", "high", "low", "volume"]]
y = df["target"]

if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    st.success(f"Model Akurasi: {accuracy:.2f}")
else:
    st.warning("Data terlalu sedikit untuk pelatihan model.")
    st.stop()

# Tampilan Data
st.subheader("\U0001F4C1 Data Terbaru")
st.dataframe(df.tail())

# Grafik Candlestick
st.subheader("\U0001F4C9 Grafik Candlestick Harga BTC/USDT")
fig = go.Figure(data=[go.Candlestick(
    x=df["timestamp"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Waktu (WIB)",
    yaxis_title="Harga (USDT)"
)
st.plotly_chart(fig, use_container_width=True)

# Prediksi Harga
last_data = X.tail(1)
prediksi = model.predict(last_data)

st.success(f"\U0001F3AF Prediksi Harga Selanjutnya: ${prediksi[0]:,.2f}")
