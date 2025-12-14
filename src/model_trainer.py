# src/model_trainer.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ==========================================
# 1. 模型定义区 (保持不变)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, nhead=4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True, dim_feedforward=hidden_size*2, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(x.size(2))
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output[:, -1, :]
        output = self.decoder(output)
        return output

# ==========================================
# 2. 训练管理器
# ==========================================
class ModelTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists('models'): os.makedirs('models')

    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def prepare_data(self, df: pd.DataFrame, target_col='Close', feature_cols=['Close'], seq_length=60):
        # 1. 构建全量处理列表
        if target_col not in feature_cols:
            process_cols = feature_cols + [target_col]
        else:
            process_cols = feature_cols
            
        # 2. 全量归一化
        data = df[process_cols].values
        self.scaler.fit(data) 
        scaled_data = self.scaler.transform(data)
        
        # 3. 记录关键索引
        self.n_features = len(process_cols) 
        self.target_col_idx = process_cols.index(target_col)
        
        # 4. 生成序列
        X_temp, y_temp = self.create_sequences(scaled_data, seq_length)
        
        # 5. 精细切割
        y = y_temp[:, self.target_col_idx]
        
        feature_indices = [process_cols.index(c) for c in feature_cols]
        X = X_temp[:, :, feature_indices]
        
        # 6. 划分数据集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return (torch.from_numpy(X_train).float().to(self.device),
                torch.from_numpy(y_train).float().to(self.device),
                torch.from_numpy(X_test).float().to(self.device),
                torch.from_numpy(y_test).float().to(self.device))

    def train(self, X_train, y_train, X_test, y_test, params, progress_callback=None):
        input_size = X_train.shape[2]
        # 【修改点1】将 input_size 存入 params，方便保存时带走
        params['input_size'] = input_size
        
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        model_type = params.get('model_type', 'LSTM')

        if model_type == 'GRU':
            self.model = GRUModel(input_size, hidden_size, num_layers).to(self.device)
        elif model_type == 'Transformer':
            self.model = TransformerModel(input_size, hidden_size, num_layers, nhead=4).to(self.device)
        else:
            self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        
        train_losses, val_losses = [], []
        
        for epoch in range(params['epochs']):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_test)
                val_loss = criterion(val_out, y_test.unsqueeze(1))
                val_losses.append(val_loss.item())
            
            if progress_callback:
                progress_callback(epoch + 1, params['epochs'], loss.item(), val_loss.item())
                
        return train_losses, val_losses

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        
        dummy = np.zeros((len(preds), self.n_features))
        dummy[:, self.target_col_idx] = preds.flatten()
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, self.target_col_idx]

    def inverse_transform_y(self, y_tensor):
        y_np = y_tensor.cpu().numpy()
        dummy = np.zeros((len(y_np), self.n_features))
        dummy[:, self.target_col_idx] = y_np.flatten()
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, self.target_col_idx]

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        correct_direction = np.sign(true_diff) == np.sign(pred_diff)
        direction_acc = np.mean(correct_direction) * 100
        return rmse, mae, r2, direction_acc
    
    # 【修改点2】升级版保存：保存参数 + 权重
    def save(self, path, params):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': params
        }, path)

    # 【修改点3】新增加载功能
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # 检查是否包含参数
        if 'params' not in checkpoint:
            raise ValueError("旧模型文件无法加载，请重新训练新模型。")
            
        params = checkpoint['params']
        
        # 重建模型结构
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        model_type = params.get('model_type', 'LSTM')
        
        if model_type == 'GRU':
            self.model = GRUModel(input_size, hidden_size, num_layers).to(self.device)
        elif model_type == 'Transformer':
            self.model = TransformerModel(input_size, hidden_size, num_layers, nhead=4).to(self.device)
        else:
            self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)
            
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        return params