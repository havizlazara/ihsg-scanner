import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cloudscraper
import json
from datetime import datetime, timedelta
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="IHSG Pro - Frequency Fix", layout="wide")

# ===============================================
# FUNGSI SCRAPING FREKUENSI (ULTIMATE VERSION)
# ===============================================

def get_idx_frequency_data(target_date):
    """Mengambil data frekuensi dengan penanganan blokir yang lebih kuat."""
    date_str = target_date.strftime('%Y%m%d')
    url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?date={date_str}&start=0&length=1000"
    
    # Inisialisasi scraper dengan browser yang lebih modern
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    
    try:
        # Menambahkan header referer agar terlihat seperti akses dari website resmi
        headers = {
            'Referer': 'https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        response = scraper.get(url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            raw_json = response.json()
            if 'data' in raw_json and raw_json['data']:
                # Mapping Kode Saham ke Frekuensi
                return {item['StockCode']: item['Frequency'] for item in raw_json['data']}
            else:
                st.sidebar.warning("Data IDX kosong untuk tanggal ini (mungkin hari libur).")
        else:
            st.sidebar.error(f"IDX memblokir akses (Status: {response.status_code})")
            
    except Exception as e:
        st.sidebar.error(f"Gagal koneksi ke server IDX: {e}")
    
    return {}

# ===============================================
# FUNGSI INDIKATOR TEKNIKAL (DI 3,3)
# ===============================================

def calculate_technical(df):
    close, high, low = df['Close'], df['High'], df['Low']
    length = 3
    
    # DI Calculation
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    up, down = high - high.shift(1), low.shift(1) - low
    
    p_di = 100 * (pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr)
    m_di = 100 * (pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr)
    
    # RSI 3
    diff = close.diff()
    gain = (diff.where(diff > 0, 0)).ewm(com=2, adjust=False).mean()
    loss = (-diff.where(diff < 0, 0)).ewm(com=2, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain/loss)))
    
    return p_di, m_di, rsi, close.ewm(span=50, adjust=False).mean()

# ===============================================
# UI DASHBOARD
# ===============================================

st.title("📈 IHSG Scanner - Real Frequency")

if 'cache_results' not in st.session_state:
    st.session_state.cache_results = None

st.sidebar.header("⚙️ Pengaturan")
# IDX data biasanya update malam hari, gunakan H-1 atau H-2
default_date = datetime.now() - timedelta(days=1)
if default_date.weekday() == 6: # Minggu
    default_date -= timedelta(days=2)
elif default_date.weekday() == 5: # Sabtu
    default_date -= timedelta(days=1)

target_date = st.sidebar.date_input("Tanggal Analisa", default_date)
btn_run = st.sidebar.button("🚀 Mulai Scan")

FILE_NAME = 'daftar_saham (2).csv'

if btn_run:
    if not os.path.exists(FILE_NAME):
        st.error(f"File {FILE_NAME} tidak ditemukan!")
    else:
        with st.spinner("Menghubungi Server IDX & Yahoo Finance..."):
            # 1. Ambil Data Frekuensi
            freq_map = get_idx_frequency_data(target_date)
            
            # 2. Baca Ticker
            tickers = pd.read_csv(FILE_NAME)['Ticker'].tolist()
            
            final_list = []
            progress = st.progress(0)
            
            for i, t in enumerate(tickers):
                try:
                    ticker = t.strip()
                    # Download Technical Data
                    data = yf.download(f"{ticker}.JK", start=target_date - timedelta(days=365), end=target_date + timedelta(days=1), progress=False)
                    
                    if not data.empty and len(data) > 10:
                        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                        
                        p_di, m_di, rsi, ema50 = calculate_technical(data)
                        curr, prev = data.iloc[-1], data.iloc[-2]
                        
                        # Sinyal Logic
                        is_cross = (p_di.iloc[-1] > m_di.iloc[-1]) and (p_di.iloc[-2] <= m_di.iloc[-2])
                        f_val = freq_map.get(ticker, 0)
                        
                        final_list.append({
                            'Ticker': ticker,
                            'Harga': int(curr['Close']),
                            '%_Change': round(((curr['Close']/prev['Close'])-1)*100, 2),
                            'Frekuensi': f_val,
                            'Sinyal_DI': "BULLISH CROSS" if is_cross else "Neutral",
                            'EMA50': "Above" if curr['Close'] > ema50.iloc[-1] else "Below",
                            'RSI_3': round(rsi.iloc[-1], 2),
                            '_cross': is_cross,
                            '_ema': curr['Close'] > ema50.iloc[-1],
                            '_rsi30': (rsi.iloc[-1] > 30 and rsi.iloc[-2] <= 30)
                        })
                except: continue
                progress.progress((i + 1) / len(tickers))
            
            st.session_state.cache_results = pd.DataFrame(final_list)

# --- FILTER HEADER ---
if st.session_state.cache_results is not None:
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1: f1 = st.checkbox("DI Cross Up Only")
    with c2: f2 = st.checkbox("Above EMA 50 Only")
    with c3: f3 = st.checkbox("RSI Cross 30 Only")
    with c4: min_f = st.number_input("Min. Frekuensi", value=0, step=100)

    res = st.session_state.cache_results.copy()
    if f1: res = res[res['_cross']]
    if f2: res = res[res['_ema']]
    if f3: res = res[res['_rsi30']]
    res = res[res['Frekuensi'] >= min_f]

    st.write(f"Menampilkan {len(res)} saham.")
    st.dataframe(res.drop(columns=['_cross', '_ema', '_rsi30']), hide_index=True, use_container_width=True)