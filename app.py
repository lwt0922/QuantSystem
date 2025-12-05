import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# 确保能找到 src 包
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import DataManager

# --- 页面配置 ---
st.set_page_config(page_title="QuantSystem Pro", layout="wide")
st.title("📈 量化分析系统 v1.0")

# --- 初始化 ---
@st.cache_resource
def get_manager():
    return DataManager()

dm = get_manager()

# --- 侧边栏：控制面板 ---
st.sidebar.header("数据控制台")
ticker_input = st.sidebar.text_input("输入股票代码", value="000001.SZ")

# 日期选择逻辑
with st.sidebar.expander("📅 时间范围设置", expanded=True):
    use_custom_date = st.checkbox("启用时间过滤", value=True, help="勾选后：\n1. 下载时只抓取该时段\n2. 展示时只显示该时段")
    
    # 默认最近一年
    default_end = pd.to_datetime("today")
    default_start = default_end - pd.Timedelta(days=365)
    
    start_input = st.date_input("开始日期", value=default_start)
    end_input = st.date_input("结束日期", value=default_end)

# 操作按钮
col1, col2 = st.sidebar.columns(2)
with col1:
    btn_update = st.button("下载/更新数据")
with col2:
    btn_load = st.button("仅加载本地")

# --- 状态管理 ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- 核心逻辑：数据获取 ---
if btn_update:
    # 1. 准备参数
    s_date = None
    e_date = None
    
    if use_custom_date:
        s_date = start_input.strftime('%Y-%m-%d')
        e_date = end_input.strftime('%Y-%m-%d')
        msg = f'正在获取 {ticker_input} ({s_date} ~ {e_date})...'
    else:
        msg = f'正在智能获取 {ticker_input} 最新数据...'

    with st.spinner(msg):
        try:
            # 2. 调用后端 (下载并存库)
            # 注意：这里返回的是库里该股票的"所有"数据
            df_all = dm.update_data(ticker_input, start_date=s_date, end_date=e_date)
            
            if df_all is not None and not df_all.empty:
                st.session_state.df = df_all
                st.success(f"✅ 操作成功！数据库现存 {len(df_all)} 条数据")
            else:
                st.warning(f"⚠️ 未找到数据，请检查代码或网络。")
        except Exception as e:
            st.error(f"发生错误: {e}")

if btn_load:
    df_all = dm.load_from_db(ticker_input)
    if df_all.empty:
        st.warning("本地数据库无此股票数据，请先点击【下载/更新数据】。")
    else:
        st.session_state.df = df_all
        st.success("本地数据加载完成。")

# --- 核心逻辑：数据展示 (View Layer) ---
if st.session_state.df is not None and not st.session_state.df.empty:
    
    # 1. 获取全量数据
    df_view = st.session_state.df.copy()
    
    # 2. 【关键修改】应用视图过滤器
    # 只有当用户勾选了时间范围，才对展示数据进行裁剪
    if use_custom_date:
        # 转换 pandas timestamp 进行比较
        dt_start = pd.to_datetime(start_input)
        dt_end = pd.to_datetime(end_input) + pd.Timedelta(days=1) #包含结束当天
        
        # 过滤数据 (Index 切片)
        df_view = df_view.loc[(df_view.index >= dt_start) & (df_view.index < dt_end)]
        
        if df_view.empty:
            st.warning(f"数据库里有数据，但在您选择的时间段 ({start_input} ~ {end_input}) 内没有数据。")
            st.stop()

    # --- 以下代码只针对 df_view (过滤后的数据) 绘图 ---
    
    latest = df_view.iloc[-1]
    prev = df_view.iloc[-2] if len(df_view) > 1 else latest
    change = latest['close'] - prev['close']
    pct_change = (change / prev['close']) * 100 if prev['close'] != 0 else 0
    
    st.markdown(f"### 市场概览 ({ticker_input})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("收盘价", f"{latest['close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    m2.metric("最高价", f"{latest['high']:.2f}")
    m3.metric("最低价", f"{latest['low']:.2f}")
    m4.metric("成交量", f"{latest['volume']:,}")

    st.markdown("### K线走势图")
    # 计算均线 (基于展示数据计算，或者基于全量计算再裁剪都可以，这里基于展示数据)
    df_view['MA20'] = df_view['close'].rolling(window=20).mean()
    
    fig = go.Figure()
    
    # K线
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view['open'], high=df_view['high'],
        low=df_view['low'], close=df_view['close'],
        name='OHLC'
    ))
    
    # 均线
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view['MA20'], 
        line=dict(color='orange', width=1), 
        name='MA 20'
    ))

    # 移除底部的 Range Slider，因为我们已经有侧边栏过滤器了
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看原始数据 (已过滤)"):
        st.dataframe(df_view.sort_index(ascending=False))

else:
    st.info("👈 请在左侧输入代码并获取数据")