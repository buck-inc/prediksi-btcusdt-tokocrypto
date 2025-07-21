import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Prediksi Harga Crypto (CoinGecko)", layout="wide")
st.title("📈 Prediksi Harga Crypto dari CoinGecko")

@st.cache_data(ttl=600)
def get_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "minutely"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            st.error("Gagal mengambil data dari CoinGecko.")
            return None
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=7)
        df["open"] = df["price"].shift(1)
        df["high"] = df["price"].rolling(5).max()
        df["low"] = df["price"].rolling(5).min()
        df["close"] = df["price"]
        df["volume"] = 0
        df["target"] = df["close"].shift(-1)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error saat ambil data: {e}")
        return None

df = get_data()
if df is None or len(df) < 10:
    st.stop()

X = df[["open", "high", "low", "volume"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
akurasi = model.score(X_test, y_test)
st.success(f"✅ Akurasi Model: {akurasi:.2f}")

st.subheader("📊 Data Terbaru")
st.dataframe(df.tail())

st.subheader("🕯️ Candlestick Chart (CoinGecko)")
fig = go.Figure(data=[go.Candlestick(
    x=df["timestamp"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(title="Grafik BTC/USD dari CoinGecko", xaxis_title="Waktu", yaxis_title="Harga")
st.plotly_chart(fig, use_container_width=True)

last_data = X.tail(1)
prediksi = model.predict(last_data)
st.success(f"🎯 Prediksi Harga Selanjutnya: ${prediksi[0]:,.2f}")