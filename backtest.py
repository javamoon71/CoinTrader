# 주요 라이브러리
import os, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
from binance.client import Client
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 한글 폰트 설정
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

# 설정
COIN_NAME = "BTC"
INTERVAL = "15m"
SEQ_LEN = 60
DAYS = 30
DATA_DIR = "D:/PythonProjects/CoinTrader/data"
MODEL_DIR = "D:/PythonProjects/CoinTrader/models"
TODAY_STR = pd.Timestamp.today().strftime('%y%m%d')

client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

# 데이터 수집
def get_ohlcv(symbol, interval, start_time):
    klines = client.get_historical_klines(symbol, interval, start_time)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'close']]
    df['close'] = df['close'].astype(float)
    return df

# 예측 함수들
def predict_lstm(model, scaler, recent_prices):
    scaled = scaler.transform(recent_prices.reshape(-1, 1))
    X = scaled[-SEQ_LEN:].reshape((1, SEQ_LEN, 1))
    pred = model.predict(X)
    return scaler.inverse_transform(pred)[0][0]

def predict_arima(model, recent_prices):
    return model.predict(n_periods=1).iloc[0]

def predict_prophet(model, current_time):
    future = pd.DataFrame({'ds': [current_time + pd.Timedelta(minutes=15)]})
    forecast = model.predict(future)
    yhat = forecast['yhat'].values[0]
    uncertainty = forecast['yhat_upper'].values[0] - forecast['yhat_lower'].values[0]
    return yhat, uncertainty

# 모델 로딩
def load_all_models():
    lstm_path    = os.path.join(MODEL_DIR, f"{COIN_NAME}USDT_{TODAY_STR}_LSTM.pkl")
    arima_path   = os.path.join(MODEL_DIR, f"{COIN_NAME}USDT_{TODAY_STR}_ARIMA.pkl")
    prophet_path = os.path.join(MODEL_DIR, f"{COIN_NAME}USDT_{TODAY_STR}_Prophet.pkl")

    lstm_models    = joblib.load(lstm_path)
    arima_models   = joblib.load(arima_path)
    prophet_models = joblib.load(prophet_path)

    return lstm_models[INTERVAL], arima_models[INTERVAL], prophet_models[INTERVAL]

# 백테스트 실행
def run_backtest():
    print("📊 멀티 모델 백테스트 시작")
    now_utc = datetime.now(timezone.utc)
    start_time = (now_utc - timedelta(days=DAYS)).strftime('%Y-%m-%d %H:%M:%S')
    df = get_ohlcv(COIN_NAME + "USDT", INTERVAL, start_time)

    lstm_info, arima_info, prophet_info = load_all_models()
    model_dict = {
        "LSTM": {
            "model": lstm_info.model,
            "scaler": lstm_info.scaler,
            "pnl": [], "pos": None, "entry": 0,
            "buy_times": [], "sell_times": []
        },
        "ARIMA": {
            "model": arima_info.model,
            "pnl": [], "pos": None, "entry": 0,
            "buy_times": [], "sell_times": []
        },
        "Prophet": {
            "model": prophet_info.model,
            "pnl": [], "pos": None, "entry": 0,
            "buy_times": [], "sell_times": []
        }
    }

    for i in range(SEQ_LEN, len(df)):
        recent_prices = df['close'].values[i-SEQ_LEN:i]
        current_price = df['close'].values[i]
        current_time = df['timestamp'].values[i]

        for name, info in model_dict.items():
            if name == "LSTM":
                pred = predict_lstm(info['model'], info['scaler'], recent_prices)
                change_rate = (pred - current_price) / current_price
                confidence = 0.0
            elif name == "ARIMA":
                pred = predict_arima(info['model'], recent_prices)
                change_rate = (pred - current_price) / current_price
                confidence = 0.0
            elif name == "Prophet":
                pred, uncertainty = predict_prophet(info['model'], pd.to_datetime(current_time))
                change_rate = (pred - current_price) / current_price
                confidence = uncertainty

            # 매매 전략 (예측 변화율 기반 + Prophet 신뢰도 필터링)
            if info['pos'] is None and change_rate > 0.005 and (name != "Prophet" or confidence < 10):
                info['pos'] = "LONG"
                info['entry'] = current_price
                info['buy_times'].append(current_time)
            elif info['pos'] == "LONG" and change_rate < -0.005:
                pnl = current_price - info['entry']
                info['pnl'].append(pnl)
                info['pos'] = None
                info['entry'] = 0
                info['sell_times'].append(current_time)

    # 성능 지표 출력
    for name, info in model_dict.items():
        pnl_array = np.array(info['pnl'])
        total = pnl_array.sum()
        trades = len(pnl_array)
        win_rate = np.mean(pnl_array > 0) if trades > 0 else 0
        avg_pnl = pnl_array.mean() if trades > 0 else 0
        max_loss = pnl_array.min() if trades > 0 else 0
        sharpe = avg_pnl / pnl_array.std() if pnl_array.std() > 0 else 0
        print(f"✅ {name} 총 수익: {total:.2f} USDT | 거래 횟수: {trades} | 승률: {win_rate:.2%} | 평균 수익: {avg_pnl:.2f} | 최대 손실: {max_loss:.2f} | 샤프 지수: {sharpe:.2f}")

    # 시각화: 시세 흐름 및 거래 시점
    plt.figure(figsize=(12, 6))
    timestamps = df['timestamp'].values[SEQ_LEN:]
    prices = df['close'].values[SEQ_LEN:]
    plt.plot(timestamps, prices, label="BTC 시세 흐름", color='gray', alpha=0.5)

    for name, info in model_dict.items():
        buy_prices = df.set_index('timestamp').loc[info['buy_times']]['close'].values
        sell_prices = df.set_index('timestamp').loc[info['sell_times']]['close'].values
        plt.scatter(info['buy_times'], buy_prices, marker='^', color='green', label=f"{name} 매수", alpha=0.6)
        plt.scatter(info['sell_times'], sell_prices, marker='v', color='red', label=f"{name} 매도", alpha=0.6)

    plt.title("📈 모델별 시세 흐름 및 거래 시점")
    plt.xlabel("시간")
    plt.ylabel("코인 시세 (USDT)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 누적 수익 시각화
    plt.figure(figsize=(10, 5))
    for name, info in model_dict.items():
        pnl_array = np.array(info['pnl'])
        cum_pnl = np.cumsum(pnl_array)
        plt.plot(cum_pnl, label=f"{name} 누적 수익")
    plt.title("📊 모델별 누적 수익 비교")
    plt.xlabel("거래 순서")
    plt.ylabel("누적 수익 (USDT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    run_backtest()
