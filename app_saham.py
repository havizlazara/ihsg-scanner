import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Bismillah Cuan", layout="wide")

# --- PARAMETER TEKNIKAL ---
DI_LENGTH = 3
DI_SMOOTHING = 3
RSI_PERIOD = 3

# ===============================================
# FUNGSI INDIKATOR DML (Directional Movement)
# ===============================================

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # 1. Perhitungan DI (Directional Indicator)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/DI_LENGTH, adjust=False).mean()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    df['plus_DI'] = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    df['minus_DI'] = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    
    # 2. RSI 3 & EMA 50
    df['RSI_3'] = (close.diff().where(close.diff() > 0, 0).ewm(com=RSI_PERIOD-1, adjust=False).mean() / 
                  abs(close.diff().where(close.diff() < 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()).pipe(lambda rs: 100 - (100 / (1 + rs)))
    
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['Pct_Change'] = close.pct_change() * 100
    
    return df

# ===============================================
# LOGIKA SESSION STATE
# ===============================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# ===============================================
# UI UTAMA
# ===============================================

st.title("🏹 Bismillah Cuan")
st.sidebar.header("📡 Kontrol Analisa")
target_date = st.sidebar.date_input("Tanggal Analisa", datetime.now())
btn_run = st.sidebar.button("Jalankan Analisa Baru")

FILE_NAME = 'daftar_saham (2).csv'

if btn_run:
    if not os.path.exists(FILE_NAME):
        st.error(f"File `{FILE_NAME}` tidak ditemukan!")
    else:
        with st.spinner("Menganalisa Sinyal DI Crossover..."):
            df_file = pd.read_csv(FILE_NAME)
            all_tickers = [str(t).strip() + ".JK" for t in df_file["Ticker"].tolist()]
            
            results = []
            progress_bar = st.progress(0)
            start_dt = target_date - timedelta(days=365)
            end_dt = target_date + timedelta(days=1)
            
            for i, ticker in enumerate(all_tickers):
                try:
                    data = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
                    if data.empty or len(data) < 50: continue
                    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                    
                    data = calculate_indicators(data)
                    curr, prev = data.iloc[-1], data.iloc[-2]
                    
                    # Logika DI Crossover (+DI memotong ke atas -DI)
                    di_cross_up = (curr['plus_DI'] > curr['minus_DI']) and (prev['plus_DI'] <= prev['minus_DI'])
                    
                    results.append({
                        'Ticker': ticker.replace('.JK', ''),
                        'Harga': round(curr['Close'], 0),
                        '%_Chg': round(curr['Pct_Change'], 2),
                        '+DI': round(curr['plus_DI'], 2),
                        '-DI': round(curr['minus_DI'], 2),
                        'DI_Signal': "BULLISH CROSS" if di_cross_up else "Netral",
                        'EMA50_Pos': "Above" if curr['Close'] > curr['EMA50'] else "Below",
                        'RSI_3': round(curr['RSI_3'], 2),
                        # Hidden Filters
                        '_di_cross': di_cross_up,
                        '_ema_above': curr['Close'] > curr['EMA50'],
                        '_rsi_30': (curr['RSI_3'] > 30 and prev['RSI_3'] <= 30)
                    })
                except: continue
                progress_bar.progress((i + 1) / len(all_tickers))
            
            st.session_state.raw_data = pd.DataFrame(results)

# --- HEADER FILTER (INSTAN Tanpa Loading) ---
if st.session_state.raw_data is not None:
    st.markdown("---")
    st.subheader("🔍 Filter Hasil Analisa")
    
    c1, c2, c3 = st.columns(3)
    with c1: f_di = st.checkbox("Hanya +DI Cross Up -DI")
    with c2: f_ema = st.checkbox("Hanya Harga > EMA 50")
    with c3: f_rsi = st.checkbox("Hanya RSI Cross Up 30")

    df_filtered = st.session_state.raw_data.copy()
    
    if f_di: df_filtered = df_filtered[df_filtered['_di_cross'] == True]
    if f_ema: df_filtered = df_filtered[df_filtered['_ema_above'] == True]
    if f_rsi: df_filtered = df_filtered[df_filtered['_rsi_30'] == True]

    st.write(f"Menampilkan **{len(df_filtered)}** saham.")
    
    # Tampilkan kolom visual
    show_cols = ['Ticker', 'Harga', '%_Chg', '+DI', '-DI', 'DI_Signal', 'EMA50_Pos', 'RSI_3']
    st.dataframe(df_filtered[show_cols], hide_index=True, use_container_width=True)
    
    csv = df_filtered[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "ihsg_di_scan.csv", "text/csv")