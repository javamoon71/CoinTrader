import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from binance.client import Client

# ─────────────────────────────────────────────
# 🔧 설정
COIN_NAME = "BTC"
INTERVAL = "15m"
SEQ_LEN = 60
DAYS = 3  # 최근 몇 일치 데이터 수집
DATA_DIR = "D:/PythonProjects/CoinTrader/data"
MODEL_DIR = "D:/PythonProjects/CoinTrader/models"
TODAY_STR = pd.Timestamp.today().strftime('%y%m%d')

# Binance API 키 설정
client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 📥 거래내역 조회 함수
def get_ohlcv(symbol, interval, start_time):
    klines = client.get_historical_klines(symbol, interval, start_time)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    return df

# 🔄 데이터 업데이트 함수
def update_data(symbol, interval, days):
    full_symbol = symbol.upper() + 'USDT'
    filename = os.path.join(DATA_DIR, f"{full_symbol}_{interval}.csv")
    start_time = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, parse_dates=['timestamp'])
        last_date = existing_df['timestamp'].max()
        start_time = last_date.strftime('%Y-%m-%d %H:%M:%S')
        new_df = get_ohlcv(full_symbol, interval, start_time)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset='timestamp').sort_values('timestamp')
    else:
        combined_df = get_ohlcv(full_symbol, interval, start_time)

    combined_df.to_csv(filename, index=False)
    return combined_df

# ─────────────────────────────────────────────
# 📦 모델 로딩
def load_models():
    lstm_path = os.path.join(MODEL_DIR, f"{COIN_NAME}USDT_{TODAY_STR}_LSTM.pkl")
    return joblib.load(lstm_path)

# 🔮 LSTM 예측
def predict_lstm(model, scaler, recent_data):
    scaled = scaler.transform(recent_data.reshape(-1, 1))
    X = scaled[-SEQ_LEN:].reshape((1, SEQ_LEN, 1))
    pred = model.predict(X)
    return scaler.inverse_transform(pred)[0][0]

# 📊 모델 성능 시각화
def plot_model_performance(model_dict, title="모델별 MSE 비교"):
    labels = list(model_dict.keys())
    mses = [model_dict[k]['mse'] for k in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, mses, color='lightgreen')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 🧪 예측 실행
def run_prediction():
    df = update_data(COIN_NAME, INTERVAL, DAYS)
    recent_prices = df['close'].values[-SEQ_LEN:]
    model_dict = load_models()
    model_info = model_dict.get(INTERVAL)

    if not model_info:
        print(f"❌ 모델 없음: {INTERVAL}")
        return

    pred_price = predict_lstm(model_info['model'], model_info['scaler'], recent_prices)
    print(f"📌 예측 가격 ({COIN_NAME}USDT, {INTERVAL}): {pred_price:.2f}")

# ─────────────────────────────────────────────
# 🌐 API 서버
app = Flask(__name__)
model_dict = load_models()

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.json.get('recent_prices')
    if not data or len(data) < SEQ_LEN:
        return jsonify({'error': f'최근 가격 데이터가 부족합니다 (최소 {SEQ_LEN}개 필요)'})

    model_info = model_dict.get(INTERVAL)
    pred = predict_lstm(model_info['model'], model_info['scaler'], np.array(data))
    return jsonify({'prediction': round(pred, 2)})

# ─────────────────────────────────────────────
# 🚀 실행
if __name__ == "__main__":
    run_prediction()
    plot_model_performance(model_dict, "LSTM 모델 MSE 비교")
    # app.run(port=5000)  # 필요 시 주석 해제