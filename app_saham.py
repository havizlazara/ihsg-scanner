import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Global Aggressive Scanner Pro", layout="wide")

# --- PARAMETER TEKNIKAL ---
DI_LENGTH = 3
RSI_PERIOD = 3

# ===============================================
# FUNGSI INDIKATOR TEKNIKAL
# ===============================================

def calculate_indicators(df):
    close, high, low = df['Close'], df['High'], df['Low']
    
    # DI Crossover (3,3)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/DI_LENGTH, adjust=False).mean()
    up_move, down_move = high - high.shift(1), low.shift(1) - low
    
    df['plus_DI'] = 100 * (pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    df['minus_DI'] = 100 * (pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index).ewm(alpha=1/DI_LENGTH, adjust=False).mean() / atr)
    
    # RSI 3 & EMA 50
    diff = close.diff()
    gain = (diff.where(diff > 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss = (-diff.where(diff < 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    df['RSI_3'] = 100 - (100 / (1 + (gain/loss)))
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['Pct_Change'] = close.pct_change() * 100
    
    return df

# ===============================================
# FUNGSI PROSES SCANNING (DENGAN PENGECEKAN GAGAL)
# ===============================================

def run_market_scan(file_name, is_indo=True, target_date=None):
    if not os.path.exists(file_name):
        st.error(f"File `{file_name}` tidak ditemukan!")
        return None, []

    df_file = pd.read_csv(file_name)
    raw_tickers = df_file["Ticker"].dropna().unique().tolist()
    
    results = []
    failed_tickers = []
    progress_bar = st.progress(0)
    
    # Gunakan rentang waktu lebih lebar untuk EMA 50
    start_dt = target_date - timedelta(days=400)
    end_dt = target_date + timedelta(days=2)

    for i, t in enumerate(raw_tickers):
        ticker = str(t).strip().upper()
        symbol = f"{ticker}.JK" if is_indo else ticker

        try:
            # threads=False menghindari masalah koneksi pada scan masal
            data = yf.download(symbol, start=start_dt, end=end_dt, progress=False, auto_adjust=True, threads=False)
            
            if data.empty:
                failed_tickers.append({"Ticker": ticker, "Alasan": "Data Kosong / Ticker Salah"})
                continue
                
            if len(data) < 50:
                failed_tickers.append({"Ticker": ticker, "Alasan": "Data < 50 hari (Baru IPO)"})
                continue
            
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(0)
            
            data = calculate_indicators(data)
            curr, prev = data.iloc[-1], data.iloc[-2]
            
            is_di_cross = (curr['plus_DI'] > curr['minus_DI']) and (prev['plus_DI'] <= prev['minus_DI'])
            is_ema50 = curr['Close'] > curr['EMA50']
            is_rsi30 = (curr['RSI_3'] > 30 and prev['RSI_3'] <= 30)
            
            results.append({
                'Ticker': ticker,
                'Harga': round(curr['Close'], 2),
                '%_Change': round(curr['Pct_Change'], 2),
                'DI_Signal': "BULLISH CROSS" if is_di_cross else "Netral",
                'EMA50': "Above" if is_ema50 else "Below",
                'RSI_3': round(curr['RSI_3'], 2),
                '+DI': round(curr['plus_DI'], 2),
                '-DI': round(curr['minus_DI'], 2),
                '_di': is_di_cross, '_ema': is_ema50, '_rsi': is_rsi30
            })
        except Exception as e:
            failed_tickers.append({"Ticker": ticker, "Alasan": str(e)})
            continue
            
        progress_bar.progress((i + 1) / len(raw_tickers))
    
    return pd.DataFrame(results), failed_tickers

# ===============================================
# UI UTAMA DENGAN TABS
# ===============================================

st.title("🌎 Global Market Aggressive Scanner")
tab_indo, tab_us = st.tabs(["🇮🇩 IHSG Market", "🇺🇸 US Market"])

st.sidebar.header("📡 Command Center")
target_date = st.sidebar.date_input("Tanggal Analisa", datetime.now())

# --- TAB INDONESIA ---
with tab_indo:
    if st.button("🚀 Jalankan Scan IHSG"):
        data_indo, fail_indo = run_market_scan('daftar_saham (2).csv', is_indo=True, target_date=target_date)
        st.session_state.indo_data = data_indo
        st.session_state.indo_fail = fail_indo

    if 'indo_data' in st.session_state and st.session_state.indo_data is not None:
        st.markdown("### Filter IHSG")
        c1, c2, c3 = st.columns(3)
        f_di = c1.checkbox("Filter DI Cross Up (IHSG)")
        f_ema = c2.checkbox("Filter Above EMA 50 (IHSG)")
        f_rsi = c3.checkbox("Filter RSI Cross 30 (IHSG)")
        
        df = st.session_state.indo_data.copy()
        if f_di: df = df[df['_di']]
        if f_ema: df = df[df['_ema']]
        if f_rsi: df = df[df['_rsi']]
        
        st.info(f"Menampilkan **{len(df)}** saham dari total {len(st.session_state.indo_data)} yang berhasil di-scan.")
        st.dataframe(df.drop(columns=['_di','_ema','_rsi']), hide_index=True, use_container_width=True)

        if st.session_state.indo_fail:
            with st.expander("⚠️ Lihat Ticker IHSG yang Gagal"):
                st.table(pd.DataFrame(st.session_state.indo_fail))

# --- TAB US MARKET ---
with tab_us:
    if st.button("🚀 Jalankan Scan US"):
        data_us, fail_us = run_market_scan('saham_us.csv', is_indo=False, target_date=target_date)
        st.session_state.us_data = data_us
        st.session_state.us_fail = fail_us

    if 'us_data' in st.session_state and st.session_state.us_data is not None:
        st.markdown("### Filter US Market")
        c1, c2, c3 = st.columns(3)
        f_di_us = c1.checkbox("Filter DI Cross Up (US)")
        f_ema_us = c2.checkbox("Filter Above EMA 50 (US)")
        f_rsi_us = c3.checkbox("Filter RSI Cross 30 (US)")
        
        df_us = st.session_state.us_data.copy()
        if f_di_us: df_us = df_us[df_us['_di']]
        if f_ema_us: df_us = df_us[df_us['_ema']]
        if f_rsi_us: df_us = df_us[df_us['_rsi']]
        
        st.info(f"Menampilkan **{len(df_us)}** saham dari total {len(st.session_state.us_data)} yang berhasil di-scan.")
        st.dataframe(df_us.drop(columns=['_di','_ema','_rsi']), hide_index=True, use_container_width=True)

        if st.session_state.us_fail:
            with st.expander("⚠️ Lihat Ticker US yang Gagal"):
                st.table(pd.DataFrame(st.session_state.us_fail))