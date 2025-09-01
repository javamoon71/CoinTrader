# training.py
import os
from dotenv import load_dotenv
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from pmdarima import auto_arima
from prophet import Prophet
from utils.model_package import ModelPackage
import sys

# warnings 무시
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_file_path = os.path.dirname(os.path.abspath(__file__))
# 프로젝트의 root 디렉토리 (부모 디렉토리)
root_dir = os.path.dirname(current_file_path)
dotenv_path = os.path.join(root_dir, '.env')
# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path)

# 설정값: 환경 변수 또는 상대 경로 사용
DATA_DIR = os.path.join(root_dir, os.getenv('DATA_DIR', 'data'))
MODEL_DIR = os.path.join(root_dir, os.getenv('MODEL_DIR', 'models'))
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🔧 데이터 디렉토리: {DATA_DIR}")
print(f"🔧 모델 디렉토리: {MODEL_DIR}")

# ============================ 기술 지표 계산 함수 (개선) ============================
def calculate_all_indicators(df):
    """
    주어진 DataFrame에 다양한 기술 지표를 추가합니다.
    """
    # 1차 보간 (선형 보간)
    df['close'] = df['close'].interpolate(method='linear')
    df['volume'] = df['volume'].interpolate(method='linear')

    # RSI (Relative Strength Index)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss.where(loss != 0, 1e-10)
        return 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    # OBV (On-Balance Volume)
    def calculate_obv(close, volume):
        obv = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv
        
    df['rsi'] = calculate_rsi(df['close'], periods=14)
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma25'] = df['close'].rolling(window=25).mean()
    df['ma99'] = df['close'].rolling(window=99).mean()
    df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    if 'volume' in df.columns:
        df['obv'] = calculate_obv(df['close'], df['volume'])
    else:
        df['obv'] = 0 # 거래량 데이터가 없을 경우 0으로 처리

    # 최종 보간 및 결측치 제거
    return df.dropna().copy()

def create_sequences(data, seq_length=60, features_to_predict_indices=[0, 1]):
    """
    LSTM 모델을 위한 시퀀스 데이터셋을 생성합니다.
    features_to_predict_indices는 예측할 피처의 인덱스입니다 (예: [0]은 'close'만, [0, 1]은 'close'와 'rsi').
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, features_to_predict_indices])
    
    return np.array(X), np.array(y)

def train_lstm(df):
    try:
        df_with_indicators = calculate_all_indicators(df.copy())
        
        # LSTM에 사용할 피처 선택 (close와 rsi는 예측에 사용되므로 반드시 포함)
        features = ['close', 'rsi', 'ma7', 'ma25', 'ma99', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'obv']
        
        # 데이터프레임에 없는 피처는 제거
        features = [f for f in features if f in df_with_indicators.columns]
        
        # 예측할 피처의 인덱스 (close, rsi)
        features_to_predict = ['close', 'rsi']
        features_to_predict_indices = [features.index(f) for f in features_to_predict]

        values = df_with_indicators[features].values
        
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        seq_length = 60
        X, y = create_sequences(scaled, seq_length, features_to_predict_indices)
        
        n_features = len(features)
        X = X.reshape((X.shape[0], seq_length, n_features))
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dense(y.shape[1])  # 출력 레이어는 예측할 값의 개수에 맞춤
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # 모델 평가
        pred = model.predict(X, verbose=0)
        
        # 역정규화를 위한 더미 배열 생성
        y_dummy = np.zeros((y.shape[0], n_features))
        y_dummy[:, features_to_predict_indices] = y
        y_rescaled = scaler.inverse_transform(y_dummy)
        
        pred_dummy = np.zeros((pred.shape[0], n_features))
        pred_dummy[:, features_to_predict_indices] = pred
        pred_rescaled = scaler.inverse_transform(pred_dummy)
        
        # close와 rsi 각각에 대한 MSE 계산
        mse_close = mean_squared_error(y_rescaled[:, features_to_predict_indices[0]], pred_rescaled[:, features_to_predict_indices[0]])
        mse_rsi = mean_squared_error(y_rescaled[:, features_to_predict_indices[1]], pred_rescaled[:, features_to_predict_indices[1]])

        return ModelPackage(
            model=model,
            scaler=scaler,
            features=features, # <-- 추가: 학습에 사용된 피처 목록
            mse_close=mse_close,
            mse_rsi=mse_rsi,
            trained_at=datetime.now()
        )
    except Exception as e:
        print(f"LSTM 학습 실패: {e}")
        return None

def train_arima(df):
    df['close'] = df['close'].interpolate(method='linear')
    series = df['close']
    model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    pred = model.predict_in_sample()
    mse_close = mean_squared_error(series, pred)

    return ModelPackage(
        model=model,
        mse_close=mse_close,
        mse_rsi=None,  # ARIMA는 RSI를 예측하지 않음
        trained_at=datetime.now()
    )

def train_prophet(df):
    df['close'] = df['close'].interpolate(method='linear')
    ps = df.reset_index()[['timestamp', 'close']].rename(columns={'timestamp':'ds','close':'y'})
    m = Prophet(daily_seasonality=True)
    m.add_country_holidays(country_name='KR')
    m.fit(ps)

    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    mse_close = mean_squared_error(ps['y'], forecast['yhat'][:len(ps)])

    return ModelPackage(
        model=m,
        mse_close=mse_close,
        mse_rsi=None,  # Prophet는 RSI를 예측하지 않음
        trained_at=datetime.now()
    )

def save_model_if_better(path, new_model_dict):
    if os.path.exists(path):
        existing = joblib.load(path)
        for tf in new_model_dict:
            # 기존 모델이 있고, MSE가 더 좋지 않다면 저장하지 않음
            if tf in existing and existing[tf] is not None and new_model_dict[tf] is not None and new_model_dict[tf].mse_close >= existing[tf].mse_close:
                print(f"🔁 기존 모델이 더 우수하여 저장 생략: {path} ({tf})")
                new_model_dict[tf] = existing[tf]
    joblib.dump(new_model_dict, path)
    print(f"✅ 모델 저장 완료: {path}")

def visualize_mse_comparison(models_dict, title):
    plt.figure(figsize=(10, 5))
    for model_type, models in models_dict.items():
        timeframes = sorted(list(models.keys()))
        mses = [models[tf].mse_close for tf in timeframes if models[tf] is not None]
        if mses:
            plt.plot(timeframes, mses, label=model_type)
    plt.title(title)
    plt.xlabel("Timeframe")
    plt.ylabel("MSE (Close)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def train_models_grouped_by_type(coin_name, timeframes):
    today_str = datetime.today().strftime('%Y%m%d')

    def process_model(model_type, train_func):
        models = {}
        for tf in timeframes:
            file_path = os.path.join(DATA_DIR, f"{coin_name}_{tf}.csv")
            if not os.path.exists(file_path):
                print(f"⚠️ 파일 없음: {file_path}")
                continue

            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            print(f"▶ {model_type} 학습 시작: {coin_name} ({tf})")
            result = train_func(df)
            models[tf] = result

        model_path = os.path.join(MODEL_DIR, f"{coin_name}_{today_str}_{model_type}.pkl")
        save_model_if_better(model_path, models)
        return models

    with ThreadPoolExecutor() as executor:
        futures = {
            'LSTM': executor.submit(process_model, 'LSTM', train_lstm),
            'ARIMA': executor.submit(process_model, 'ARIMA', train_arima),
            'Prophet': executor.submit(process_model, 'Prophet', train_prophet)
        }

        results = {name: future.result() for name, future in futures.items()}
        visualize_mse_comparison(results, f"{coin_name} 모델별 MSE 비교")

if __name__ == "__main__":
    timeframes = ['15m', '1h', '4h', '1d']
    train_models_grouped_by_type("BTCUSDT", timeframes)
    train_models_grouped_by_type("ETHUSDT", timeframes)
    train_models_grouped_by_type("XRPUSDT", timeframes)