import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(page_title="Prediksi Harga Crypto Tokocrypto", layout="wide")
st.title("📈 Prediksi Harga Crypto (Data Binance/Tokocrypto)")

@st.cache_data
def get_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 1000}  # 1 menit terakhir
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "_1", "_2", "_3", "_4", "_5", "_6"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta")  # 👉 Ubah ke WIB
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["target"] = df["close"].shift(-1)

    return df.dropna()

df = get_data()

if df is None or len(df) < 10:
    st.error("❌ Gagal mengambil data atau data terlalu sedikit.")
    st.stop()

# Model
X = df[["open", "high", "low", "volume"]]
y = df["target"]

if len(X) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    akurasi = model.score(X_test, y_test)
    st.success(f"Model Akurasi: {akurasi:.2f}")
else:
    st.warning("⚠️ Data terlalu sedikit untuk pelatihan model.")
    st.stop()

# Tampilan Data Terbaru
st.subheader("📊 Data Terbaru")
st.dataframe(df.tail())

# Prediksi Harga Selanjutnya
last_data = X.tail(1)
prediksi = model.predict(last_data)
st.success(f"🎯 Prediksi Harga Selanjutnya: ${prediksi[0]:,.2f}")
