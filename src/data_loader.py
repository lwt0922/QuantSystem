import yfinance as yf
import akshare as ak
import pandas as pd
import sqlite3
import os
import time
import random
import requests
from datetime import datetime

class DataManager:
    def __init__(self, db_path='data/quant_data.db'):
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_db()
        
        # --- 伪装配置 ---
        # 常见浏览器 User-Agent 列表
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67"
        ]

    def _ensure_data_dir(self):
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        ''')
        conn.commit()
        conn.close()

    # --- 伪装核心功能 ---
    
    def _get_random_headers(self):
        """生成随机请求头，伪装成浏览器"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def _random_sleep(self, min_seconds=1.5, max_seconds=3.5):
        """
        随机等待，模仿人类操作频率
        避免触发服务器的 Rate Limit (速率限制)
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        # print(f"⏳ 正在模拟人类思考，暂停 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)

    # ------------------

    def fetch_data(self, ticker, start_date=None, end_date=None):
        """
        混合获取逻辑 (带伪装)
        """
        ticker = ticker.strip().upper()
        print(f"🕵️ 正在请求数据: {ticker} (Start: {start_date})")
        
        # 每次请求前随机暂停，极其重要！
        self._random_sleep()

        try:
            if ticker.endswith('.SS') or ticker.endswith('.SZ'):
                return self._fetch_from_akshare(ticker, start_date, end_date)
            else:
                return self._fetch_from_yahoo(ticker, start_date, end_date)
        except Exception as e:
            print(f"❌ 数据获取发生严重错误: {e}")
            return None

    def _fetch_from_akshare(self, ticker, start_date, end_date):
        print(">>> [A股] 切换至 AkShare 引擎...")
        code = ticker.split('.')[0]
        
        if not start_date:
            s_date = "20200101"
        else:
            s_date = start_date.replace("-", "")
            
        if not end_date:
            e_date = datetime.now().strftime("%Y%m%d")
        else:
            e_date = end_date.replace("-", "")

        try:
            # AkShare 内部已经封装了 headers，但我们通过外层的 sleep 降低了被封概率
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
            
            if df is None or df.empty:
                print("⚠️ AkShare 未返回数据。")
                return None
            
            rename_map = {
                '日期': 'date', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            return df
            
        except Exception as e:
            print(f"AkShare 报错: {e}")
            return None

    def _fetch_from_yahoo(self, ticker, start_date, end_date):
        print(">>> [美股/国际] 切换至 Yahoo Finance 引擎...")
        
        # 配置带有伪装 Headers 的 Session
        session = requests.Session()
        session.headers.update(self._get_random_headers())

        try:
            # 将伪装的 session 传给 yfinance (部分版本支持)
            # 注意：Yahoo 对 IP 封锁很敏感，如果依然报错，通常是 IP 问题而非 Header 问题
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                session=session,  # 注入伪装 Session
                timeout=10        # 设置超时
            )
        except TypeError:
            # 兼容不支持 session 参数的旧版本
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
            
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' not in df.columns:
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df.columns = [str(c).lower() for c in df.columns]
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [c for c in required_cols if c in df.columns]
        return df[final_cols]

    def save_to_db(self, ticker, df):
        if df is None or df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        df_to_save = df.copy()
        df_to_save['ticker'] = ticker
        
        data = df_to_save.to_dict('records')
        c = conn.cursor()
        c.executemany('''
            INSERT OR IGNORE INTO stock_prices (ticker, date, open, high, low, close, volume)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        ''', data)
        conn.commit()
        conn.close()
        print(f"✅ {ticker} 数据已存入数据库 (共 {len(df)} 条)")

    def load_from_db(self, ticker):
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM stock_prices WHERE ticker = '{ticker}' ORDER BY date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric)
        return df

    def update_data(self, ticker, start_date=None, end_date=None):
        """
        智能更新逻辑：
        1. 如果指定了 start_date，则强制下载指定时间段的数据 (Custom Mode)
        2. 如果未指定，则自动检测本地进度进行增量更新 (Smart Mode)
        """
        ticker = ticker.upper().strip()
        
        # --- 模式 1: 用户指定时间段 (强制下载) ---
        if start_date is not None:
            print(f"🔧 [强制模式] 用户指定下载: {ticker} ({start_date} ~ {end_date})")
            df_new = self.fetch_data(ticker, start_date=start_date, end_date=end_date)
            
            if df_new is not None and not df_new.empty:
                self.save_to_db(ticker, df_new)
                return self.load_from_db(ticker)
            else:
                # 如果下载失败，尝试返回本地已有数据
                return self.load_from_db(ticker)

        # --- 模式 2: 智能增量更新 (默认) ---
        df_local = self.load_from_db(ticker)
        
        smart_start_date = "2020-01-01"
        if not df_local.empty:
            last_date = df_local.index[-1]
            smart_start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📥 [智能模式] 本地已有数据，增量更新起点: {smart_start_date}")
        else:
            print(f"📥 [智能模式] 本地无数据，默认从 {smart_start_date} 开始全量下载")
        
        today = datetime.today().strftime('%Y-%m-%d')
        
        # 如果不需要更新 (smart_start_date 已经是明天或未来)
        if smart_start_date > today:
             print("✅ 数据已是最新，无需更新。")
             return df_local

        df_new = self.fetch_data(ticker, start_date=smart_start_date)
        
        if df_new is not None and not df_new.empty:
            self.save_to_db(ticker, df_new)
            return self.load_from_db(ticker)
        else:
            return df_local