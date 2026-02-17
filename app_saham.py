import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="IHSG Instan Filter", layout="wide")

# --- PARAMETER TEKNIKAL ---
RSI_PERIOD = 3
ADX_SMOOTHING = 3
DI_LENGTH = 3

# ===============================================
# FUNGSI INDIKATOR
# ===============================================

def calculate_indicators(df):
    close = df['Close']
    df['Pct_Change'] = close.pct_change() * 100
    
    # RSI 3
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rs = gain / loss
    df['RSI_3'] = 100 - (100 / (1 + rs))
    
    # ADX 3
    high, low = df['High'], df['Low']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/DI_LENGTH, adjust=False).mean()
    up, down = high - high.shift(1), low.shift(1) - low
    plus_di = 100 * (pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100)
    df['ADX_3'] = dx.ewm(alpha=1/ADX_SMOOTHING, adjust=False).mean()
    
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    return df

# ===============================================
# LOGIKA SESSION STATE (AGAR TIDAK ANALISA ULANG)
# ===============================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# ===============================================
# UI STREAMLIT
# ===============================================

st.title("🚀 IHSG Fast Filter Dashboard")
st.write("Analisa dilakukan sekali, filter bisa dilakukan berkali-kali secara instan.")

# --- SIDEBAR: HANYA UNTUK ANALISA UTAMA ---
st.sidebar.header("📡 Download Data")
target_date = st.sidebar.date_input("Tanggal Analisa", datetime.now())
btn_run = st.sidebar.button("Jalankan Analisa Baru")

FILE_NAME = 'daftar_saham (2).csv'

# PROSES DOWNLOAD (Hanya jalan jika tombol diklik)
if btn_run:
    if not os.path.exists(FILE_NAME):
        st.error(f"File `{FILE_NAME}` tidak ditemukan!")
    else:
        with st.spinner("Sedang mengambil data dari Yahoo Finance..."):
            df_file = pd.read_csv(FILE_NAME)
            all_tickers = [str(t).strip() + ".JK" for t in df_file["Ticker"].tolist()]
            
            results = []
            progress_bar = st.progress(0)
            start_dt = target_date - timedelta(days=500)
            end_dt = target_date + timedelta(days=1)
            
            for i, ticker in enumerate(all_tickers):
                try:
                    data = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
                    if data.empty or len(data) < 50: continue
                    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                    
                    data = calculate_indicators(data)
                    latest, prev = data.iloc[-1], data.iloc[-2]
                    
                    is_ema50 = latest['Close'] > latest['EMA50']
                    is_rsi_cross = (latest['RSI_3'] > 30 and prev['RSI_3'] <= 30)
                    is_adx_rising = latest['ADX_3'] > prev['ADX_3']
                    
                    results.append({
                        'Ticker': ticker.replace('.JK', ''),
                        'Harga': round(latest['Close'], 0),
                        '%_Change': round(latest['Pct_Change'], 2),
                        'Sinyal_RSI': "CROSS UP 30" if is_rsi_cross else ("CROSS DOWN" if latest['RSI_3'] < 30 and prev['RSI_3'] >= 30 else "Netral"),
                        'Trend_ADX': "Rising" if is_adx_rising else "Falling",
                        'EMA50': "Above" if is_ema50 else "Below",
                        'RSI_Val': round(latest['RSI_3'], 2),
                        'ADX_Val': round(latest['ADX_3'], 2),
                        # Hidden columns for filtering
                        '_ema': is_ema50,
                        '_rsi': is_rsi_cross,
                        '_adx': is_adx_rising
                    })
                except: continue
                progress_bar.progress((i + 1) / len(all_tickers))
            
            st.session_state.raw_data = pd.DataFrame(results)
            st.success("Analisa selesai! Silakan gunakan filter di bawah.")

# --- AREA FILTER (HEADER) ---
# Filter ini hanya muncul jika data sudah ada di session state
if st.session_state.raw_data is not None:
    st.markdown("---")
    st.subheader("🎯 Filter Hasil Analisa (Tanpa Loading Ulang)")
    
    # Membuat 3 kolom untuk filter di bagian atas tabel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        f_ema = st.checkbox("Hanya Harga > EMA 50")
    with col2:
        f_rsi = st.checkbox("Hanya RSI Cross Up 30")
    with col3:
        f_adx = st.checkbox("Hanya ADX Rising")

    # Logika Filter Instan
    df_filtered = st.session_state.raw_data.copy()
    
    if f_ema:
        df_filtered = df_filtered[df_filtered['_ema'] == True]
    if f_rsi:
        df_filtered = df_filtered[df_filtered['_rsi'] == True]
    if f_adx:
        df_filtered = df_filtered[df_filtered['_adx'] == True]

    # Menampilkan Tabel
    st.write(f"Menampilkan **{len(df_filtered)}** saham dari total {len(st.session_state.raw_data)}")
    
    # Buat kolom yang ingin ditampilkan (buang kolom bantuan _)
    view_cols = ['Ticker', 'Harga', '%_Change', 'Sinyal_RSI', 'Trend_ADX', 'EMA50', 'RSI_Val', 'ADX_Val']
    
    st.dataframe(
        df_filtered[view_cols],
        hide_index=True,
        use_container_width=True
    )
    
    # Download Button
    csv = df_filtered[view_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", csv, "filtered_ihsg.csv", "text/csv")
else:
    st.info("Silakan klik tombol 'Jalankan Analisa Baru' di sidebar untuk memulai.")