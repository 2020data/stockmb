# -*- coding: utf-8 -*-
"""
============================================================
📈 綜合策略分析系統：均線 + RSI + KD + 線性迴歸 + 二次微分
功能：動態讀取、K線重採樣、回測引擎、Lightweight Charts 多圖層
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_lightweight_charts import renderLightweightCharts

# ─────────────────────────────────────────────────────────────
# 🌟 快取資料載入與自動格式偵測
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_resample_data(file_path, rule):
    # 讀取 CSV
    raw = pd.read_csv(file_path)
    
    # 自動偵測日期欄位
    time_col = 'Datetime' if 'Datetime' in raw.columns else 'Date'
    if time_col not in raw.columns:
        # 如果都不是，嘗試取第一欄
        time_col = raw.columns[0]
        
    raw[time_col] = pd.to_datetime(raw[time_col])
    raw = raw.set_index(time_col).sort_index()
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in raw.columns for col in required_cols):
        raise ValueError(f"CSV 檔案必須包含以下欄位: {required_cols}")
        
    # 重採樣邏輯
    if rule is not None:
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        if 'Volume' in raw.columns:
            ohlc_dict['Volume'] = 'sum'
        raw = raw.resample(rule).agg(ohlc_dict).dropna()
        
    return raw, time_col

# ─────────────────────────────────────────────────────────────
# ① 技術指標計算 (包含斜率 M、截距 B、二次微分)
# ─────────────────────────────────────────────────────────────
def compute_indicators(df, ma_short=20, ma_long=60, rsi_period=14,
                       k_period=9, d_period=3, slope_window=20):
    df = df.copy()

    # 1. 基礎均線
    df[f'MA{ma_short}'] = df['Close'].rolling(ma_short).mean()
    df[f'MA{ma_long}']  = df['Close'].rolling(ma_long).mean()

    # 2. RSI
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. KD
    low_min  = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    rsv      = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K']   = rsv.ewm(alpha=1/d_period, adjust=False).mean()
    df['D']   = df['K'].ewm(alpha=1/d_period, adjust=False).mean()

    # 4. 線性迴歸：計算斜率(M) 與 截距(B)
    x = np.arange(slope_window)
    x_mean = x.mean()
    Sxx = np.sum((x - x_mean)**2)

    def calc_slope(y):
        if len(y) < slope_window: return np.nan
        return np.sum((x - x_mean) * (y - np.mean(y))) / Sxx

    df['Slope_m'] = df['Close'].rolling(slope_window).apply(calc_slope, raw=True)
    # B = average(y) - M * average(x)
    df['Intercept_b'] = df['Close'].rolling(slope_window).mean() - df['Slope_m'] * x_mean

    # 5. 二次微分 (Second Derivative / 加速度)
    # f''(x) = f(x) - 2f(x-1) + f(x-2)
    df['Second_Diff'] = df['Close'].diff().diff()

    return df

# ─────────────────────────────────────────────────────────────
# ② 訊號產生與回測引擎 (保留原有邏輯)
# ─────────────────────────────────────────────────────────────
def generate_signals(df, ma_short, ma_long, rsi_low, rsi_high, kd_low, kd_high, min_hold):
    df = df.copy()
    ma_s, ma_l = f'MA{ma_short}', f'MA{ma_long}'
    n = len(df)
    signal = [0] * n
    pos = False
    last_i = -min_hold

    for i in range(max(ma_long, 14, 9) + 1, n):
        if any(pd.isna(df[col].iloc[i]) for col in [ma_s, ma_l, 'RSI', 'K', 'D']):
            continue

        ma_s_v, ma_l_v = df[ma_s].iloc[i], df[ma_l].iloc[i]
        rsi = df['RSI'].iloc[i]
        k, k_prev = df['K'].iloc[i], df['K'].iloc[i-1]
        d, d_prev = df['D'].iloc[i], df['D'].iloc[i-1]
        cool_ok = (i - last_i) >= min_hold

        if not pos:
            if ma_s_v > ma_l_v and rsi_low < rsi < 55 and (k > d and k_prev <= d_prev) and cool_ok:
                signal[i], pos, last_i = 1, True, i
        else:
            if ((rsi > rsi_high and k > kd_high) or (ma_s_v < ma_l_v) or (k < d and k_prev >= d_prev and rsi < 50)) and cool_ok:
                signal[i], pos, last_i = -1, False, i

    df['Signal'] = signal
    return df

def backtest(df, initial_capital):
    df = df.copy()
    capital, shares, buy_price = initial_capital, 0, 0
    port_vals, trades = [], []
    comm, tax = 0.001425, 0.003

    for i, row in df.iterrows():
        if row['Signal'] == 1:
            buy_price = row['Close']
            shares = int(capital / (buy_price * (1 + comm)))
            capital -= shares * buy_price * (1 + comm)
            trades.append({'Date': i, 'Type': 'Buy', 'Price': buy_price, 'Shares': shares})
        elif row['Signal'] == -1 and shares > 0:
            capital += shares * row['Close'] * (1 - comm - tax)
            trades.append({'Date': i, 'Type': 'Sell', 'Price': row['Close'], 'Shares': shares, 'Return%': (row['Close']-buy_price)/buy_price*100})
            shares = 0
        port_vals.append(capital + shares * row['Close'])

    df['Portfolio'] = port_vals
    df['BuyHold'] = initial_capital * (df['Close'] / df['Close'].iloc[0])
    return df, pd.DataFrame(trades)

# ─────────────────────────────────────────────────────────────
# ③ Lightweight Charts 渲染引擎 (新增二次微分支援)
# ─────────────────────────────────────────────────────────────
def render_dynamic_lw_charts(df, ma_short, ma_long, config):
    def prep_data(series, name="value"):
        temp = series.dropna().reset_index()
        temp.columns = ['time', name]
        temp['time'] = temp['time'].apply(lambda x: int(x.timestamp()))
        return temp.to_dict('records')

    # 主圖：K線與訊號
    kline_df = df[['Open', 'High', 'Low', 'Close']].dropna().reset_index()
    kline_df.columns = ['time', 'open', 'high', 'low', 'close']
    kline_df['time'] = kline_df['time'].apply(lambda x: int(x.timestamp()))
    
    markers = []
    for t, row in df[df['Signal'] != 0].iterrows():
        markers.append({
            "time": int(t.timestamp()),
            "position": "belowBar" if row['Signal']==1 else "aboveBar",
            "color": "#e91e63" if row['Signal']==1 else "#4caf50",
            "shape": "arrowUp" if row['Signal']==1 else "arrowDown",
            "text": "Buy" if row['Signal']==1 else "Sell"
        })

    is_dark = config.get("theme") == "Dark"
    chart_layout = {
        "layout": {"textColor": "#d1d4dc" if is_dark else "#191919", "background": {"type": "solid", "color": "#131722" if is_dark else "#ffffff"}},
        "grid": {"vertLines": {"color": "#363c4e"}, "horzLines": {"color": "#363c4e"}},
    }

    # 組合圖層
    main_series = [{"type": "Candlestick", "data": kline_df.to_dict('records'), "markers": markers}]
    if config.get("show_ma"):
        main_series.append({"type": "Line", "data": prep_data(df[f'MA{ma_short}']), "options": {"color": "#ff9800", "title": f"MA{ma_short}"}})
        main_series.append({"type": "Line", "data": prep_data(df[f'MA{ma_long}']), "options": {"color": "#2196f3", "title": f"MA{ma_long}"}})

    charts = [{"chartOptions": {**chart_layout, "height": config.get("main_height")}, "series": main_series}]

    sub_h = config.get("sub_height")
    if config.get("show_rsi"):
        charts.append({"chartOptions": {**chart_layout, "height": sub_h}, "series": [{"type": "Line", "data": prep_data(df['RSI']), "options": {"color": "#ab47bc", "title": "RSI"}}]})
    
    if config.get("show_reg"):
        charts.append({"chartOptions": {**chart_layout, "height": sub_h, "rightPriceScale": {"visible": True}, "leftPriceScale": {"visible": True}}, 
                       "series": [
                           {"type": "Line", "data": prep_data(df['Slope_m']), "options": {"color": "#ffee58", "title": "Slope(M)", "priceScaleId": "right"}},
                           {"type": "Line", "data": prep_data(df['Intercept_b']), "options": {"color": "#8d6e63", "title": "Intercept(B)", "priceScaleId": "left"}}
                       ]})
    
    if config.get("show_diff2"):
        charts.append({"chartOptions": {**chart_layout, "height": sub_h}, 
                       "series": [{"type": "Histogram", "data": prep_data(df['Second_Diff']), "options": {"color": "#FFD600", "title": "二次微分 (加速度)"}}]})

    if config.get("show_port"):
        charts.append({"chartOptions": {**chart_layout, "height": sub_h}, "series": [{"type": "Line", "data": prep_data(df['Portfolio']), "options": {"color": "#ff7043", "title": "Portfolio"}}]})

    renderLightweightCharts(charts, 'combined_chart')

# ─────────────────────────────────────────────────────────────
# ④ Streamlit UI 主程式
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nasdaq 策略分析系統", layout="wide")

with st.sidebar:
    st.title("⚙️ 參數設定")
    folder_path = st.text_input("資料夾路徑", value="./futures_historical_data")
    
    selected_file = None
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        selected_file = st.selectbox("選擇檔案", csv_files) if csv_files else None
    
    resample_rule = st.selectbox("時間週期轉換", ["不轉換", "5分K", "15分K", "30分K", "60分K", "日K"])
    rule_map = {"不轉換": None, "5分K": "5T", "15分K": "15T", "30分K": "30T", "60分K": "60T", "日K": "D"}
    
    st.divider()
    st.subheader("👁️ 顯示控制")
    config = {
        "theme": st.selectbox("主題", ["Dark", "Light"]),
        "main_height": st.slider("主圖高度", 300, 600, 400),
        "sub_height": st.slider("副圖高度", 100, 300, 150),
        "show_ma": st.toggle("顯示均線", True),
        "show_rsi": st.toggle("顯示 RSI", True),
        "show_reg": st.toggle("顯示 斜率/截距", True),
        "show_diff2": st.toggle("顯示 二次微分", True),
        "show_port": st.toggle("顯示 資金曲線", True)
    }

    st.divider()
    slope_n = st.slider("線性迴歸週期 (N)", 5, 100, 20)
    ma_s = st.slider("短期均線", 5, 50, 20)
    ma_l = st.slider("長期均線", 30, 200, 60)

if selected_file:
    file_full_path = os.path.join(folder_path, selected_file)
    try:
        raw_df, time_col_name = load_and_resample_data(file_full_path, rule_map[resample_rule])
        
        df = compute_indicators(raw_df, ma_s, ma_l, slope_window=slope_n)
        df = generate_signals(df, ma_s, ma_l, 35, 68, 25, 75, 7)
        df, trades_df = backtest(df, 500000)

        st.title(f"📊 分析報告: {selected_file}")
        
        # 顯示指標圖表
        render_dynamic_lw_charts(df, ma_s, ma_l, config)
        
        # 顯示數據表格
        st.subheader("📋 數據明細 (最後10筆)")
        st.dataframe(df[[ 'Close', 'Slope_m', 'Intercept_b', 'Second_Diff', 'Portfolio']].tail(10))
        
        if not trades_df.empty:
            st.subheader("📜 最近交易記錄")
            st.table(trades_df.tail(5))
            
    except Exception as e:
        st.error(f"錯誤: {e}")
else:
    st.info("請在左側選擇有效的 CSV 檔案路徑。")
