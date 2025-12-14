# src/backtester.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class Backtester:
    """
    量化交易策略回测引擎 (升级版：增加信号过滤阈值)
    """
    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital 
        self.commission = commission

    def run_backtest(self, df, predictions, threshold=0.001):
        """
        执行回测
        :param threshold: 买入阈值 (例如 0.001 代表预测涨幅超过 0.1% 才买入)
        """
        # 1. 数据对齐
        min_len = min(len(df), len(predictions))
        backtest_df = df.iloc[-min_len:].copy().reset_index(drop=True)
        preds = predictions[-min_len:].flatten()
        
        backtest_df['Predicted_Close'] = preds
        
        # 2. 计算预期收益率
        # (预测价 - 现价) / 现价
        backtest_df['Expected_Return'] = (backtest_df['Predicted_Close'] - backtest_df['Close']) / backtest_df['Close']
        
        # 3. 生成交易信号 (加入阈值过滤！)
        # 只有当 预期收益率 > 阈值 时，才买入 (Signal=1)
        # 否则空仓 (Signal=0)
        # 💡 改进点：这里加了 threshold，过滤掉噪音
        backtest_df['Signal'] = np.where(backtest_df['Expected_Return'] > threshold, 1, 0)
        
        # 4. 确定持仓
        backtest_df['Position'] = backtest_df['Signal'].shift(1).fillna(0)
        
        # 5. 收益计算
        backtest_df['Market_Return'] = backtest_df['Close'].pct_change()
        backtest_df['Trade_Action'] = backtest_df['Position'].diff().abs().fillna(0)
        
        backtest_df['Strategy_Return'] = backtest_df['Position'] * backtest_df['Market_Return'] 
        backtest_df['Strategy_Return'] -= backtest_df['Trade_Action'] * self.commission
        
        # 6. 资金曲线
        backtest_df['Market_Value'] = self.initial_capital * (1 + backtest_df['Market_Return']).cumprod().fillna(self.initial_capital)
        backtest_df['Strategy_Value'] = self.initial_capital * (1 + backtest_df['Strategy_Return']).cumprod().fillna(self.initial_capital)
        
        # 修正起点
        backtest_df.loc[0, 'Market_Value'] = self.initial_capital
        backtest_df.loc[0, 'Strategy_Value'] = self.initial_capital
        
        return backtest_df

    def calculate_metrics(self, df):
        # ... (保持不变) ...
        total_return = (df['Strategy_Value'].iloc[-1] / self.initial_capital) - 1
        days = len(df)
        if days == 0: return {}
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        risk_free_rate = 0.02
        daily_returns = df['Strategy_Return']
        if daily_returns.std() == 0:
            sharpe_ratio = 0
        else:
            excess_return = daily_returns.mean() * 252 - risk_free_rate
            volatility = daily_returns.std() * (252 ** 0.5)
            sharpe_ratio = excess_return / (volatility + 1e-9)
            
        roll_max = df['Strategy_Value'].cummax()
        drawdown = df['Strategy_Value'] / roll_max - 1
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": f"{total_return*100:.2f}%",
            "Annual Return": f"{annual_return*100:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown*100:.2f}%"
        }

    def plot_results(self, df):
        # ... (保持不变) ...
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Strategy_Value'], mode='lines', name='AI 策略净值', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Market_Value'], mode='lines', name='基准 (买入持有)', line=dict(color='gray', dash='dot')))
        fig.update_layout(title="策略回测资金曲线", xaxis_title="日期", yaxis_title="账户净值", template="plotly_dark", height=500)
        return fig