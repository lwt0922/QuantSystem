# src/storage.py
import sqlite3
import pandas as pd
import os

class DataStorage:
    """
    负责数据的本地存储与读取 (SQLite)
    """
    def __init__(self, db_path='data/quant_data.db'):
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

    def save_to_db(self, df: pd.DataFrame, table_name='stock_history'):
        """
        将 DataFrame 保存到数据库，若存在则追加
        """
        if df.empty:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            # if_exists='append' 表示追加，'replace' 表示覆盖
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            return True
        except Exception as e:
            print(f"存储失败: {e}")
            return False

    def load_from_db(self, ticker: str = None, table_name='stock_history') -> pd.DataFrame:
        """
        从数据库读取数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {table_name}"
            if ticker:
                query += f" WHERE Ticker = '{ticker}'"
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            # 转换日期列格式
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            return df
        except Exception as e:
            print(f"读取失败: {e}")
            return pd.DataFrame()

    def clear_data(self, table_name='stock_history'):
        """
        清空数据表（慎用）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False