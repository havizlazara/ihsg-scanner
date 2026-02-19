import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cloudscraper
from datetime import datetime, timedelta
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="IHSG Scanner Fix", layout="wide")

# --- PARAMETER TEKNIKAL ---
DI_LENGTH = 3
RSI_PERIOD = 3

# ===============================================
# FUNGSI SCRAPING FREKUENSI (FIXED METHOD)
# ===============================================

def get_idx_frequency_fix(target_date):
    """Mengambil data frekuensi menggunakan cloudscraper untuk menembus proteksi."""
    date_str = target_date.strftime('%Y%m%d')
    url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?date={date_str}&start=0&length=1000"
    
    # Membuat scraper yang bisa melewati proteksi bot
    scraper = cloudscraper.create_scraper()
    
    try:
        response = scraper.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data and 'data' in data:
                return {item['StockCode']: item['Frequency'] for item in data['data']}
        else:
            st.sidebar.warning(f"IDX Status: {response.status_code}. Mencoba data alternatif...")
    except Exception as e:
        st.sidebar.error(f"Error IDX: {e}")
    return {}

# ===============================================
# FUNGSI INDIKATOR TEKNIKAL
# ===============================================

def calculate_indicators(df):
    close, high, low = df['Close'], df['High'], df['Low']
    
    # 1. DI Crossover (3,3)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/DI_LENGTH, adjust=False).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_di = 100 * (pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    
    # 2. RSI 3 & EMA 50
    diff = close.diff()
    gain = (diff.where(diff > 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss = (-diff.where(diff < 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain/loss)))
    
    return plus_di, minus_di, rsi, close.ewm(span=50, adjust=False).mean()

# ===============================================
# UI UTAMA
# ===============================================

st.title("🏹 IHSG Pro Scanner (Frequency Fix)")

if 'final_data' not in st.session_state:
    st.session_state.final_data = None

# SIDEBAR
st.sidebar.header("📡 Command Center")
target_date = st.sidebar.date_input("Tanggal Analisa", datetime.now() - timedelta(days=1))
btn_run = st.sidebar.button("Jalankan Scan Full")

FILE_NAME = 'daftar_saham (2).csv'

if btn_run:
    if not os.path.exists(FILE_NAME):
        st.error(f"File `{FILE_NAME}` tidak ditemukan!")
    else:
        with st.spinner("Menarik data frekuensi & teknikal..."):
            # 1. Ambil Frekuensi
            freq_dict = get_idx_frequency_fix(target_date)
            
            # 2. Baca CSV
            df_file = pd.read_csv(FILE_NAME)
            tickers = [t.strip() for t in df_file["Ticker"].tolist()]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                try:
                    data = yf.download(f"{ticker}.JK", start=target_date - timedelta(days=365), end=target_date + timedelta(days=1), progress=False)
                    if data.empty: continue
                    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

                    p_di, m_di, rsi, ema50 = calculate_indicators(data)
                    curr, prev = data.iloc[-1], data.iloc[-2]
                    
                    is_di_cross = (p_di.iloc[-1] > m_di.iloc[-1]) and (p_di.iloc[-2] <= m_di.iloc[-2])
                    freq_val = freq_dict.get(ticker, 0)

                    results.append({
                        'Ticker': ticker,
                        'Harga': int(curr['Close']),
                        '%_Change': round(((curr['Close']/prev['Close'])-1)*100, 2),
                        'Frekuensi': freq_val,
                        'DI_Signal': "BULLISH CROSS" if is_di_cross else "Netral",
                        'EMA50': "Above" if curr['Close'] > ema50.iloc[-1] else "Below",
                        'RSI_3': round(rsi.iloc[-1], 2),
                        '_di_cross': is_di_cross,
                        '_ema_above': curr['Close'] > ema50.iloc[-1],
                        '_rsi_30': (rsi.iloc[-1] > 30 and rsi.iloc[-2] <= 30)
                    })
                except: continue
                progress_bar.progress((i + 1) / len(tickers))
            
            st.session_state.final_data = pd.DataFrame(results)

# ===============================================
# FILTER HEADER
# ===============================================

if st.session_state.final_data is not None:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1: f_di = st.checkbox("Hanya +DI Cross Up")
    with col2: f_ema = st.checkbox("Hanya Harga > EMA 50")
    with col3: f_rsi = st.checkbox("Hanya RSI Cross Up 30")
    with col4: min_freq = st.number_input("Min. Frekuensi", value=0, step=500)

    df = st.session_state.final_data.copy()
    if f_di: df = df[df['_di_cross']]
    if f_ema: df = df[df['_ema_above']]
    if f_rsi: df = df[df['_rsi_30']]
    df = df[df['Frekuensi'] >= min_freq]

    st.dataframe(df.drop(columns=['_di_cross', '_ema_above', '_rsi_30']), 
                 hide_index=True, use_container_width=True)