# app.py
import os

# --- 1. 网络代理与 Tushare 修复 ---
os.environ['NO_PROXY'] = '*'
import requests
requests.Session.trust_env = False

# --- 2. 【核心修复】解决 Streamlit 监视 PyTorch 报错 ---
# ⚠️ 这段代码必须放在 import streamlit 之前
import torch
try:
    # 强行给 torch.classes 属性打补丁，骗过 Streamlit 的监视器
    if not hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = []
except:
    pass

import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 导入自定义模块 ---
from src.data_loader import DataLoader
from src.storage import DataStorage
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.backtester import Backtester

# --- 初始化模块 ---
loader = DataLoader()
storage = DataStorage()
engineer = FeatureEngineer() # 👈 初始化特征工程模块

# --- 页面配置 ---
st.set_page_config(page_title="量化分析系统 v1.1", layout="wide")
st.title("📊 量化分析系统")

# --- 侧边栏导航 ---
menu = ["数据管理与特征工程", "模型训练", "策略回测"]
choice = st.sidebar.selectbox("功能导航", menu)

if choice == "数据管理与特征工程":
    
    # 创建两列布局
    col1, col2 = st.columns([1, 2.5]) # 右侧稍微宽一点用于画图

    with col1:
        st.header("1. 数据操作区")
        
        # --- A. 数据下载 ---
        with st.expander("步骤 1: 数据下载", expanded=True):
            st.info("💡 提示：平安银行 `000001`，茅台 `600519`")
            ticker = st.text_input("输入代码", value="600519") 
            start_date = st.date_input("开始日期", datetime.date(2023, 1, 1))
            end_date = st.date_input("结束日期", datetime.date.today())
            
            if st.button("🚀 下载原始数据"):
                with st.spinner('正在抓取数据...'):
                    df = loader.fetch_data(ticker, str(start_date), str(end_date))
                    if df is not None and not df.empty:
                        st.session_state['current_data'] = df
                        # 清除可能存在的旧特征数据
                        if 'processed_data' in st.session_state:
                            del st.session_state['processed_data']
                        st.success(f"成功获取 {len(df)} 条数据！")
                    else:
                        st.error("获取失败，请检查代码或网络。")

        # --- B. 特征工程 ---
        with st.expander("步骤 2: 特征工程 (计算指标)", expanded=True):
            if 'current_data' in st.session_state:
                st.write("点击下方按钮计算 MA, RSI, MACD 等指标：")
                if st.button("⚡ 一键生成技术指标"):
                    raw_df = st.session_state['current_data']
                    
                    # 调用特征工程模块
                    processed_df = engineer.add_technical_indicators(raw_df)
                    
                    # 同时也生成预测目标 (Target)
                    processed_df = engineer.add_prediction_target(processed_df)
                    
                    st.session_state['processed_data'] = processed_df
                    st.success(f"特征生成完毕！当前列数: {processed_df.shape[1]}")
                    st.write("新增列:", list(processed_df.columns[-8:])) # 显示最后几个新增列名
            else:
                st.warning("请先在上方下载数据。")

        # --- C. 数据入库 ---
        with st.expander("步骤 3: 存储数据"):
            if st.button("💾 保存处理后的数据到数据库"):
                # 优先保存处理过的数据，如果没有则保存原始数据
                data_to_save = st.session_state.get('processed_data', st.session_state.get('current_data'))
                
                if data_to_save is not None:
                    # ✅ 修改后的代码（强制存入 feature_data，这样训练模块就能读到了）
                    success = storage.save_to_db(data_to_save, table_name='feature_data')
                    if success:
                        st.success("数据已存入 SQLite！")
                    else:
                        st.error("存储失败。")
                else:
                    st.warning("没有可保存的数据。")

    with col2:
        st.header("2. 可视化分析区")
        
        tab1, tab2 = st.tabs(["📊 K线与指标预览", "💾 数据库记录"])
        
        with tab1:
            # 优先显示处理过的数据(包含指标)，否则显示原始数据
            df_viz = st.session_state.get('processed_data', st.session_state.get('current_data'))
            
            if df_viz is not None and not df_viz.empty:
                # 使用 Plotly Subplots 绘制更专业的图 (上图K线，下图RSI)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, subplot_titles=(f'{ticker} 价格与均线', 'RSI 强弱指标'),
                                    row_width=[0.2, 0.7])

                # 1. 主图：K线
                fig.add_trace(go.Candlestick(x=df_viz['Date'],
                                open=df_viz['Open'], high=df_viz['High'],
                                low=df_viz['Low'], close=df_viz['Close'], name='K线'), row=1, col=1)

                # 2. 主图：均线 (如果有)
                if 'MA5' in df_viz.columns:
                    fig.add_trace(go.Scatter(x=df_viz['Date'], y=df_viz['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
                if 'MA20' in df_viz.columns:
                    fig.add_trace(go.Scatter(x=df_viz['Date'], y=df_viz['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
                
                # 3. 主图：布林带 (如果有)
                if 'BB_Upper' in df_viz.columns:
                    fig.add_trace(go.Scatter(x=df_viz['Date'], y=df_viz['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_viz['Date'], y=df_viz['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Lower Band'), row=1, col=1)

                # 4. 副图：RSI (如果有)
                if 'RSI' in df_viz.columns:
                    fig.add_trace(go.Scatter(x=df_viz['Date'], y=df_viz['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
                    # 添加 70/30 超买超卖线
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("### 数据明细 (前10行)")
                st.dataframe(df_viz.head(10), use_container_width=True)
            
            else:
                st.info("👈 请在左侧先下载数据，然后点击“生成技术指标”")

        # --- 修改 app.py 中的 tab2 部分 ---
        with tab2:
            st.markdown("### 本地数据库记录")
            
            # 1. 增加一个下拉框，让系统知道你要看哪张表
            # 这样就不会去读不存在的 'stock_history' 了
            table_source = st.selectbox(
                "选择数据表", 
                ["feature_data (含特征数据)", "raw_data (原始数据)"]
            )
            
            # 根据选择决定表名
            current_table = 'feature_data' if 'feature' in table_source else 'raw_data'
            
            if st.button("🔄 刷新数据库视图"):
                # 【关键修正】这里明确指定 table_name，不再使用默认的 'stock_history'
                db_data = storage.load_from_db(table_name=current_table)
                
                if db_data is not None and not db_data.empty:
                    st.write(f"表 `{current_table}` 中共有 {len(db_data)} 条记录")
                    # 按日期降序显示
                    st.dataframe(db_data.sort_values(by='Date', ascending=False).head(50), use_container_width=True)
                else:
                    st.warning(f"表 `{current_table}` 为空或不存在。请先在左侧下载并保存数据。")
            
            # 数据清理功能
            with st.expander("⚠️ 危险操作"):
                if st.button("清空所有数据库数据"):
                    storage.clear_data(table_name='raw_data')
                    storage.clear_data(table_name='feature_data')
                    st.success("数据库已清空")


elif choice == "模型训练": # 👈 把这里改成 "模型训练"
    st.header("🧠 深度学习模型训练实验室")

    # 初始化训练器
    if 'trainer' not in st.session_state:
        st.session_state['trainer'] = ModelTrainer()
    trainer = st.session_state['trainer']

    # --- 1. 数据准备区 ---
    st.subheader("1. 训练数据准备")
    
    # 自动从数据库加载数据
    db_data = storage.load_from_db(table_name='feature_data')
    if db_data is None or db_data.empty:
        st.warning("⚠️ 数据库中没有特征数据 (feature_data)。请先去“数据管理”页面下载并生成特征。")
    else:
        tickers = db_data['Ticker'].unique()
        selected_ticker = st.selectbox("选择要训练的股票", tickers)
        
        # 筛选该股票数据
        df_train = db_data[db_data['Ticker'] == selected_ticker].sort_values(by='Date')
        st.write(f"已加载 {selected_ticker} 数据，共 {len(df_train)} 条。")
        
        # 特征选择
        all_cols = [c for c in df_train.columns if c not in ['Date', 'Ticker', 'Target']]
        # 默认选中 Close, MA5, RSI (如果有)
        default_cols = ['Close']
        if 'MA5' in all_cols: default_cols.append('MA5')
        if 'RSI' in all_cols: default_cols.append('RSI')
        
        feature_cols = st.multiselect("选择输入特征 (Input Features)", all_cols, default=default_cols)
        target_col = st.selectbox("选择预测目标 (Target)", ['Close'], disabled=True, help="目前默认预测收盘价")

        st.markdown("---")

        # ... (前文代码不变) ...

        # --- 2. 模型参数配置 (侧边栏) ---
        st.sidebar.subheader("⚙️ 模型超参数设置")
        
        # 训练模式选择
        train_mode = st.sidebar.radio(
            "训练模式", 
            ["重新训练新模型", "加载已有模型"] # 👈 去掉“暂不支持”
        )
        
        # === 模式 A: 重新训练 ===
        if train_mode == "重新训练新模型":
            model_type_label = st.sidebar.selectbox(
                "模型类型", 
                ["LSTM (长短期记忆网络)", "GRU (门控循环单元)", "Transformer (注意力机制)"]
            )
            if "LSTM" in model_type_label: model_type = "LSTM"
            elif "GRU" in model_type_label: model_type = "GRU"
            elif "Transformer" in model_type_label: model_type = "Transformer"
            
            epoch_num = st.sidebar.slider("训练轮次 (Epochs)", 10, 200, 50)
            batch_size = st.sidebar.slider("批大小 (Batch Size)", 16, 128, 32)
            learning_rate = st.sidebar.number_input("学习率", 0.0001, 0.01, 0.001, format="%.4f")
            
            st.sidebar.markdown("---")
            seq_length = st.sidebar.slider("序列长度", 5, 60, 20)
            hidden_size = st.sidebar.slider("隐藏层大小", 16, 128, 64, step=4)
            num_layers = st.sidebar.slider("网络层数", 1, 4, 1)

        # === 模式 B: 加载模型 ===
        else:
            st.sidebar.info("📂 从 models/ 目录加载")
            # 扫描 models 文件夹
            if not os.path.exists('models'): os.makedirs('models')
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
            
            if len(model_files) == 0:
                st.sidebar.warning("暂无保存的模型")
                selected_model_file = None
            else:
                selected_model_file = st.sidebar.selectbox("选择模型文件", model_files)

        # --- 3. 训练/加载 控制区 ---
        st.subheader("2. 模型训练与监控")
        
        col1, col2 = st.columns([1, 3])
        
        # 逻辑分流
        if train_mode == "重新训练新模型":
            with col1:
                start_train = st.button("🔥 开始训练", type="primary")
                st.caption(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

            if start_train:
                if len(feature_cols) == 0:
                    st.error("请至少选择一个特征！")
                else:
                    with st.spinner("正在预处理数据..."):
                        # 1. 准备数据
                        X_train, y_train, X_test, y_test = trainer.prepare_data(
                            df_train, target_col='Close', feature_cols=feature_cols, seq_length=seq_length
                        )
                    
                    # 2. 进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart_placeholder = st.empty()
                    loss_history = {"train": [], "val": []}
                    
                    def update_ui(epoch, total_epochs, train_loss, val_loss):
                        progress = epoch / total_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch}/{total_epochs} - Loss: {train_loss:.5f}")
                        loss_history['train'].append(train_loss)
                        loss_history['val'].append(val_loss)
                        chart_placeholder.line_chart(pd.DataFrame(loss_history))

                    # 3. 训练
                    params = {
                        "model_type": model_type,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "learning_rate": learning_rate,
                        "epochs": epoch_num
                    }
                    
                    try:
                        trainer.train(X_train, y_train, X_test, y_test, params, progress_callback=update_ui)
                        st.success("🎉 训练完成！")
                        
                        # 保存到 Session
                        st.session_state['trained_model_params'] = params
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        
                        # 保存文件 (现在会带上 params)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                        save_path = f"models/{selected_ticker}_{model_type}_{timestamp}.pth"
                        trainer.save(save_path, params) # 👈 传入 params
                        st.toast(f"模型已保存至 {save_path}")
                        
                    except Exception as e:
                        st.error(f"训练出错: {e}")
                        import traceback
                        st.text(traceback.format_exc())

        else: # 加载模式
            with col1:
                load_btn = st.button("📂 加载模型", type="primary")
            
            if load_btn and selected_model_file:
                try:
                    load_path = os.path.join("models", selected_model_file)
                    # 调用 trainer.load
                    loaded_params = trainer.load(load_path)
                    
                    st.success(f"已加载模型: {selected_model_file}")
                    
                    # 为了能进行验证，我们还需要生成测试数据 X_test
                    with st.spinner("正在准备验证数据..."):
                        # ⚠️ 注意：这里假设你想在当前选中的股票和特征上验证模型
                        # 如果特征数量不对，程序可能会报错
                        _, _, X_test_new, y_test_new = trainer.prepare_data(
                            df_train, target_col='Close', feature_cols=feature_cols, 
                            seq_length=20 # 默认长度，或者你可以从 loaded_params 里取如果存了的话
                        )
                    
                    st.session_state['trained_model_params'] = loaded_params
                    st.session_state['X_test'] = X_test_new
                    st.session_state['y_test'] = y_test_new
                    
                except ValueError as ve:
                    st.error(f"加载失败: {ve}")
                    st.warning("提示：旧版模型文件无法加载，请先使用新版代码【重新训练】生成一个新模型文件。")
                except Exception as e:
                    st.error(f"错误: {e}")
                    import traceback
                    st.text(traceback.format_exc())

        # ... (后续验证代码不变) ...

        # --- 4. 验证结果分析 (升级版) ---
        st.markdown("---")
        st.subheader("3. 验证结果分析")
        
        if 'trained_model_params' in st.session_state:
            # 自动进行预测和评估
            if st.button("📊 生成详细评估报告"):
                X_test = st.session_state['X_test']
                y_test_tensor = st.session_state['y_test']
                
                # 1. 预测与反归一化
                # 预测值 (Predicted)
                y_pred_real = trainer.predict(X_test)
                # 真实值 (Actual)
                y_true_real = trainer.inverse_transform_y(y_test_tensor)
                
                # 2. 计算指标
                rmse, mae, r2, direction_acc = trainer.evaluate(y_true_real, y_pred_real)
                
                # 3. 显示核心指标 (使用 Streamlit 的 Metric 组件)
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("RMSE (均方根误差)", f"{rmse:.2f}", help="越小越好，表示预测值偏离真实值的程度")
                col_m2.metric("MAE (平均绝对误差)", f"{mae:.2f}", help="越小越好，表示平均偏差金额")
                col_m3.metric("R² Score (拟合度)", f"{r2:.2f}", help="越接近 1 越好，表示模型解释了多少波动")
                col_m4.metric("方向准确率", f"{direction_acc:.2f}%", help="预测涨跌方向的正确率")
                
                # 4. 绘制对比图 (Plotly)
                fig_res = go.Figure()
                
                # 真实值曲线
                fig_res.add_trace(go.Scatter(
                    y=y_true_real, 
                    mode='lines', 
                    name='真实价格 (Actual)',
                    line=dict(color='gray', width=2, dash='dot')
                ))
                
                # 预测值曲线
                fig_res.add_trace(go.Scatter(
                    y=y_pred_real, 
                    mode='lines', 
                    name='模型预测 (Predicted)',
                    line=dict(color='cyan', width=2)
                ))
                
                fig_res.update_layout(
                    title=f"{selected_ticker} 预测对比图 (验证集)",
                    xaxis_title="时间步 (Time Step)",
                    yaxis_title="价格 (Price)",
                    height=500,
                    template="plotly_dark" # 使用暗色主题，看起来更专业
                )
                st.plotly_chart(fig_res, use_container_width=True)
                
                st.success(f"✅ 评估完成！模型方向预测准确率: {direction_acc:.2f}%")
                
        else:
            st.info("请先在上方点击“🔥 开始训练”")
# ... (模型训练模块代码结束) ...

elif choice == "策略回测":
    st.header("📈 策略回测与评估")
    
    # 1. 检查状态
    if 'trained_model_params' not in st.session_state or 'trainer' not in st.session_state:
        st.warning("⚠️ 请先在“模型训练”页面训练或加载一个模型，并确保数据已准备好。")
        st.info("模型训练成功后，模型和测试数据将自动保存在 Session 中。")
    else:
        st.success("✅ 检测到已加载的模型，可以开始回测。")
        
        col1, col2 = st.columns([1, 3])
        
        # 回测参数设置
        with col1:
            st.subheader("回测参数")
            initial_capital = st.number_input("初始资金", value=100000, step=10000)
            commission = st.number_input("交易手续费 (例如 0.0003)", value=0.0003, format="%.4f")
            
            threshold = st.number_input("买入阈值 (例如 0.005 代表预测涨0.5%才买)", value=0.001, step=0.001, format="%.4f")

            run_backtest_btn = st.button("🚀 开始回测", type="primary")
            st.markdown("---")
            st.caption("回测基于模型测试集上的预测结果。")
            
        if run_backtest_btn:
            try:
                trainer = st.session_state['trainer']
                
                # 检查测试数据是否完整
                if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
                    st.error("测试数据缺失，请重新运行模型训练。")
                else:
                    X_test = st.session_state['X_test']
                    
                    with st.spinner("正在使用模型生成交易信号..."):
                        # 1. 模型预测 (生成未来价格预测)
                        predicted_prices = trainer.predict(X_test)
                    
                    # 2. 获取对应的原始行情数据 (用于计算收益)
                    # 重新从数据库加载，并对齐日期
                    db_data = storage.load_from_db(table_name='feature_data')
                    current_ticker = st.session_state.get('selected_ticker', db_data['Ticker'].iloc[0]) 
                    df_raw = db_data[db_data['Ticker'] == current_ticker].sort_values(by='Date')
                    
                    # 3. 运行回测
                    st.info("正在执行交易模拟...")
                    bt = Backtester(initial_capital=initial_capital, commission=commission)
                    result_df = bt.run_backtest(df_raw, predicted_prices, threshold=threshold) # 👈 传入 threshold

                    
                    # 4. 显示结果
                    with col2:
                        st.subheader("回测资金曲线")
                        
                        # 画图
                        fig = bt.plot_results(result_df)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("核心绩效指标")
                        
                        # 计算指标
                        metrics = bt.calculate_metrics(result_df)
                        
                        # 展示 4 个核心指标
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("总收益率", metrics["Total Return"])
                        m2.metric("年化收益", metrics["Annual Return"])
                        m3.metric("夏普比率", metrics["Sharpe Ratio"])
                        m4.metric("最大回撤", metrics["Max Drawdown"])
                        
                        with st.expander("📝 每日交易明细"):
                            st.dataframe(result_df[['Date', 'Close', 'Predicted_Close', 'Signal', 'Position', 'Strategy_Value']], use_container_width=True)
                            
            except Exception as e:
                st.error(f"回测出错: {e}")
                import traceback
                st.text(traceback.format_exc())

