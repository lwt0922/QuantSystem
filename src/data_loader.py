# src/data_loader.py
import tushare as ts
import pandas as pd
import datetime

# ==============================
# 🔴 请在这里填入你的 Token
MY_TOKEN = 'cc71d0a4718e3e4b388fedd89b2b3dc5d1fca609aab2ffac4036bebb' 
# ==============================

class DataLoader:
    def __init__(self):
        # 初始化 Tushare Pro 接口
        ts.set_token(MY_TOKEN)
        self.pro = ts.pro_api()

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        print(f"🕵️ [Tushare] 正在请求: {ticker}")
        
        # 格式转换：2023-01-01 -> 20230101
        ts_start = start_date.replace('-', '')
        ts_end = end_date.replace('-', '')
        
        # 代码转换：输入 600519 -> 自动转 600519.SH
        # Tushare 要求后缀: .SH (上交所), .SZ (深交所)
        ts_code = ticker
        if ticker.isdigit():
            if ticker.startswith('6'): ts_code = f"{ticker}.SH"
            elif ticker.startswith('0') or ticker.startswith('3'): ts_code = f"{ticker}.SZ"
        
        try:
            # 获取日线行情
            df = self.pro.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
            
            if df.empty:
                print("未获取到数据，可能是Token无效或代码错误")
                return pd.DataFrame()

            # Tushare 返回的数据是倒序的（最新日期在最前），需要反转
            df = df.iloc[::-1].reset_index(drop=True)

            # 重命名列以适配系统
            df.rename(columns={
                'trade_date': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'vol': 'Volume'
            }, inplace=True)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df['Ticker'] = ticker
            
            # 只要这些列
            return df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            print(f"❌ Tushare 报错: {e}")
            return pd.DataFrame()