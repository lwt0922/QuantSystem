# src/feature_engineer.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    特征工程模块：负责计算技术指标 (MA, RSI, MACD, Bollinger Bands) 
    并为机器学习模型准备标签 (Label)
    """
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一次性添加所有常用的技术指标
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. 移动平均线 (Trend)
        # 短期均线 (5日)，长期均线 (20日)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 2. 相对强弱指数 RSI (Momentum) - 14日
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (Trend Following Momentum)
        # EMA12 - EMA26
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 4. 布林带 (Volatility)
        # 中轨 = 20日均线，上下轨 = 中轨 ± 2 * 标准差
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (2 * std_dev)
        
        # 5. 清洗数据 (去除因计算指标产生的 NaN 空值)
        # 例如 MA20 会导致前 19 天没有数据
        df.dropna(inplace=True)
        
        return df

    def add_prediction_target(self, df: pd.DataFrame, shift_days=-1) -> pd.DataFrame:
        """
        [为未来模型准备] 添加预测目标列 (Label)
        默认为：预测“明天”的涨跌。
        Target = 1 (涨), 0 (跌)
        """
        if df.empty: return df
        df = df.copy()
        
        # 计算未来的收益率
        # shift(-1) 把明天的数据往上拉一行，对齐到今天
        df['Next_Close'] = df['Close'].shift(shift_days)
        
        # 定义目标：如果明天的收盘价 > 今天的收盘价，标记为 1，否则为 0
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
        
        # 去除最后一行 (因为它没有明天的数据)
        df.dropna(inplace=True)
        
        return df