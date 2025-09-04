import os
import logging
from dotenv import load_dotenv
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from pmdarima import auto_arima
from prophet import Prophet
import optuna
from tqdm import tqdm
import json
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# warnings 무시
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_file_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_file_path)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# 설정 파일 로드
CONFIG_PATH = os.path.join(root_dir, 'config.json')
default_config = {
    "data_dir": "data",
    "model_dir": "models",
    "timeframes": ["15m", "1h", "4h", "1d"],
    "coins": ["BTCUSDT"],
    "lstm_features": ["close", "rsi", "ma7", "ma25", "macd", "macd_signal"],
    "seq_lengths": {"15m": 15, "1h": 30, "4h": 45, "1d": 60},
    "rsi_epsilon": 1e-10,
    "max_workers": max(1, os.cpu_count() // 2)
}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.warning(f"config.json 파일이 없습니다. 기본 설정을 사용합니다: {CONFIG_PATH}")
    config = default_config

DATA_DIR = os.path.join(root_dir, os.getenv('DATA_DIR', config.get('data_dir', 'data')))
MODEL_DIR = os.path.join(root_dir, os.getenv('MODEL_DIR', config.get('model_dir', 'models')))
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info(f"🔧 데이터 디렉토리: {DATA_DIR}")
logger.info(f"🔧 모델 디렉토리: {MODEL_DIR}")

# ModelPackage 클래스 정의
@dataclass
class ModelPackage:
    model: any
    scaler: any = None
    features: list = None
    mse_close: float = None
    mse_rsi: float = None
    mae_close: float = None
    rmse_close: float = None
    dir_acc_close: float = None
    trained_at: datetime = None

# 데이터 검증 함수
def validate_data(df, min_rows=100):
    if len(df) < min_rows:
        raise ValueError(f"데이터 길이가 너무 짧습니다: {len(df)} 행, 최소 {min_rows} 행 필요")
    if (df['close'] <= 0).any() or ('volume' in df and (df['volume'] < 0).any()):
        logger.warning("음수 가격 또는 거래량 감지, 제거 중...")
        df = df[df['close'] > 0]
        if 'volume' in df:
            df = df[df['volume'] >= 0]
    return df

# 기술 지표 계산 함수
def calculate_all_indicators(df, rsi_period=14, ma_periods=[7, 25], macd_params=(12, 26, 9)):
    df['close'] = df['close'].interpolate(method='time')
    df['volume'] = df['volume'].interpolate(method='time') if 'volume' in df else 0

    def calculate_rsi(data, periods):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss.where(loss != 0, config.get('rsi_epsilon', 1e-10))
        return 100 - (100 / (1 + rs))

    def calculate_macd(data, fast_period, slow_period, signal_period):
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    for period in ma_periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    df['macd'], df['macd_signal'] = calculate_macd(df['close'], *macd_params)
    df['golden_cross'] = ((df['ma25'].shift(1) <= df['ma25'].shift(1)) & (df['ma25'] > df['ma25'])).astype(int)
    df['dead_cross'] = ((df['ma25'].shift(1) >= df['ma25'].shift(1)) & (df['ma25'] < df['ma25'])).astype(int)

    return df.dropna().copy()

# 시퀀스 생성
def create_sequences(data, seq_length, features_to_predict_indices):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length, features_to_predict_indices])
    return np.array(X), np.array(y)

# 방향 정확도 계산
def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true[1:] - y_true[:-1])
    direction_pred = np.sign(y_pred[1:] - y_pred[:-1])
    return np.mean(direction_true == direction_pred)

# LSTM 하이퍼파라미터 튜닝
def optimize_lstm(trial, X_train, y_train, seq_length, n_features, features_to_predict_indices):
    units = trial.suggest_int('units', 16, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(seq_length, n_features)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(len(features_to_predict_indices))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model

# LSTM 학습
def train_lstm(df, timeframe):
    try:
        df = validate_data(df)
        df_with_indicators = calculate_all_indicators(df.copy())
        features = config.get('lstm_features', ['close', 'rsi', 'ma7', 'ma25', 'macd', 'macd_signal'])
        features = [f for f in features if f in df_with_indicators.columns]
        features_to_predict = ['close', 'rsi']
        features_to_predict_indices = [features.index(f) for f in features_to_predict if f in features]

        values = df_with_indicators[features].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        seq_length = config.get('seq_lengths', {'15m': 15, '1h': 30, '4h': 45, '1d': 60}).get(timeframe, 30)
        X, y = create_sequences(scaled, seq_length, features_to_predict_indices)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        n_features = len(features)
        X_train = X_train.reshape((X_train.shape[0], seq_length, n_features))
        X_test = X_test.reshape((X_test.shape[0], seq_length, n_features))

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optimize_lstm(trial, X_train, y_train, seq_length, n_features, features_to_predict_indices).evaluate(X_test, y_test), n_trials=5)
        best_model = optimize_lstm(study.best_trial, X_train, y_train, seq_length, n_features, features_to_predict_indices)

        pred = best_model.predict(X_test, verbose=0)
        y_dummy = np.zeros((y_test.shape[0], n_features))
        y_dummy[:, features_to_predict_indices] = y_test
        y_rescaled = scaler.inverse_transform(y_dummy)
        pred_dummy = np.zeros((pred.shape[0], n_features))
        pred_dummy[:, features_to_predict_indices] = pred
        pred_rescaled = scaler.inverse_transform(pred_dummy)

        mse_close = mean_squared_error(y_rescaled[:, features_to_predict_indices[0]], pred_rescaled[:, features_to_predict_indices[0]])
        mse_rsi = mean_squared_error(y_rescaled[:, features_to_predict_indices[1]], pred_rescaled[:, features_to_predict_indices[1]])
        mae_close = mean_absolute_error(y_rescaled[:, features_to_predict_indices[0]], pred_rescaled[:, features_to_predict_indices[0]])
        rmse_close = np.sqrt(mse_close)
        dir_acc_close = directional_accuracy(y_rescaled[:, features_to_predict_indices[0]], pred_rescaled[:, features_to_predict_indices[0]])

        return ModelPackage(
            model=best_model,
            scaler=scaler,
            features=features,
            mse_close=mse_close,
            mse_rsi=mse_rsi,
            mae_close=mae_close,
            rmse_close=rmse_close,
            dir_acc_close=dir_acc_close,
            trained_at=datetime.now()
        )
    except ValueError as e:
        logger.error(f"LSTM 학습 실패 (ValueError): {e}")
        return None
    except Exception as e:
        logger.error(f"LSTM 학습 실패: {e}")
        return None

# ARIMA 학습
def train_arima(df, timeframe):
    try:
        df = validate_data(df)
        series = df['close']
        exog = df[['volume', 'rsi']] if 'volume' in df and 'rsi' in df else None
        model = auto_arima(series, seasonal=True, m=7, exogenous=exog, stepwise=True, suppress_warnings=True, max_p=3, max_q=3)
        
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        exog_train = exog[:train_size] if exog is not None else None
        exog_test = exog[train_size:] if exog is not None else None
        
        model.fit(train, exog=exog_train)
        pred = model.predict(n_periods=len(test), exogenous=exog_test)
        mse_close = mean_squared_error(test, pred)
        mae_close = mean_absolute_error(test, pred)
        rmse_close = np.sqrt(mse_close)
        dir_acc_close = directional_accuracy(test.values, pred)

        return ModelPackage(
            model=model,
            mse_close=mse_close,
            mse_rsi=None,
            mae_close=mae_close,
            rmse_close=rmse_close,
            dir_acc_close=dir_acc_close,
            trained_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"ARIMA 학습 실패: {e}")
        return None

# Prophet 학습
def train_prophet(df, timeframe):
    try:
        df = validate_data(df)
        ps = df.reset_index()[['timestamp', 'close', 'volume', 'rsi']].rename(columns={'timestamp': 'ds', 'close': 'y'})
        m = Prophet(daily_seasonality=False, weekly_seasonality=True)
        m.add_country_holidays(country_name='KR')
        if 'volume' in ps:
            m.add_regressor('volume')
        if 'rsi' in ps:
            m.add_regressor('rsi')
        
        train_size = int(len(ps) * 0.8)
        train, test = ps[:train_size], ps[train_size:]
        m.fit(train)
        
        future = m.make_future_dataframe(periods=len(test))
        future['volume'] = ps['volume']
        future['rsi'] = ps['rsi']
        forecast = m.predict(future)
        
        mse_close = mean_squared_error(test['y'], forecast['yhat'][train_size:])
        mae_close = mean_absolute_error(test['y'], forecast['yhat'][train_size:])
        rmse_close = np.sqrt(mse_close)
        dir_acc_close = directional_accuracy(test['y'].values, forecast['yhat'][train_size:].values)

        return ModelPackage(
            model=m,
            mse_close=mse_close,
            mse_rsi=None,
            mae_close=mae_close,
            rmse_close=rmse_close,
            dir_acc_close=dir_acc_close,
            trained_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"Prophet 학습 실패: {e}")
        return None

# 백테스팅 함수
def backtest_predictions(model_package, df, timeframe, features_to_predict_indices=[0, 1]):
    df = validate_data(df)
    df_with_indicators = calculate_all_indicators(df.copy())
    features = model_package.features if hasattr(model_package, 'features') else ['close']
    scaler = model_package.scaler if hasattr(model_package, 'scaler') else None
    
    if model_package.model.__class__.__name__ == 'Sequential':
        values = df_with_indicators[features].values
        scaled = scaler.transform(values)
        seq_length = config.get('seq_lengths', {'15m': 15, '1h': 30, '4h': 45, '1d': 60}).get(timeframe, 30)
        X, _ = create_sequences(scaled, seq_length, features_to_predict_indices)
        X = X.reshape((X.shape[0], seq_length, len(features)))
        pred = model_package.model.predict(X, verbose=0)
        pred_dummy = np.zeros((pred.shape[0], len(features)))
        pred_dummy[:, features_to_predict_indices] = pred
        pred_rescaled = scaler.inverse_transform(pred_dummy)
        predictions = pred_rescaled[:, features_to_predict_indices[0]]
    else:
        predictions = model_package.model.predict(n_periods=len(df)) if model_package.model.__class__.__name__ == 'ARIMA' else model_package.model.predict(df.reset_index()[['ds']])['yhat'].values

    signals = np.where(df['golden_cross'] == 1, 1, np.where(df['dead_cross'] == 1, -1, 0))
    returns = df['close'].pct_change().shift(-1)
    strategy_returns = signals * returns
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
    
    return {'sharpe_ratio': sharpe_ratio, 'returns': strategy_returns.cumsum()}

# 앙상블 예측
def ensemble_predictions(models_dict, df, timeframe):
    predictions = []
    weights = {'LSTM': 0.5, 'ARIMA': 0.25, 'Prophet': 0.25}
    for model_type, models in models_dict.items():
        if timeframe in models and models[timeframe]:
            model_package = models[timeframe]
            if model_package.model.__class__.__name__ == 'Sequential':
                values = calculate_all_indicators(df.copy())[model_package.features].values
                scaler = model_package.scaler
                scaled = scaler.transform(values)
                seq_length = config.get('seq_lengths', {'15m': 15, '1h': 30, '4h': 45, '1d': 60}).get(timeframe, 30)
                X, _ = create_sequences(scaled, seq_length, [0])
                X = X.reshape((X.shape[0], seq_length, len(model_package.features)))
                pred = model_package.model.predict(X, verbose=0)
                pred_dummy = np.zeros((pred.shape[0], len(model_package.features)))
                pred_dummy[:, 0] = pred[:, 0]
                pred_rescaled = scaler.inverse_transform(pred_dummy)[:, 0]
                predictions.append(pred_rescaled * weights[model_type])
            else:
                pred = model_package.model.predict(n_periods=len(df)) if model_package.model.__class__.__name__ == 'ARIMA' else model_package.model.predict(df.reset_index()[['ds']])['yhat'].values
                predictions.append(pred * weights[model_type])
    
    return np.sum(predictions, axis=0) / sum(weights.values())

# 모델 저장
def save_model_if_better(path, new_model_dict):
    try:
        if os.path.exists(path):
            existing = joblib.load(path)
            for tf in new_model_dict:
                if tf in existing and existing[tf] and new_model_dict[tf]:
                    weighted_score = (new_model_dict[tf].mse_close * 0.7 + new_model_dict[tf].rmse_close * 0.3) if new_model_dict[tf].mse_close else float('inf')
                    existing_score = (existing[tf].mse_close * 0.7 + existing[tf].rmse_close * 0.3) if existing[tf].mse_close else float('inf')
                    if weighted_score >= existing_score:
                        logger.info(f"🔁 기존 모델이 더 우수하여 저장 생략: {path} ({tf})")
                        new_model_dict[tf] = existing[tf]
        joblib.dump(new_model_dict, path)
        logger.info(f"✅ 모델 저장 완료: {path}")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")

# 시각화
def visualize_results(models_dict, coin_name, df, timeframe):
    plt.figure(figsize=(12, 8))
    metrics = ['mse_close', 'mae_close', 'rmse_close', 'dir_acc_close']
    for metric in metrics:
        plt.subplot(2, 2, metrics.index(metric) + 1)
        for model_type, models in models_dict.items():
            timeframes = sorted(list(models.keys()))
            values = [getattr(models[tf], metric) for tf in timeframes if models[tf] and getattr(models[tf], metric) is not None]
            if values:
                plt.plot(timeframes, values, label=model_type)
        plt.title(f"{coin_name} - {metric.upper()}")
        plt.xlabel("타임프레임")
        plt.ylabel(metric.upper())
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{coin_name}_metrics_comparison.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    for model_type, models in models_dict.items():
        if timeframe in models and models[timeframe]:
            model_package = models[timeframe]
            if model_package.model.__class__.__name__ == 'Sequential':
                values = calculate_all_indicators(df.copy())[model_package.features].values
                scaler = model_package.scaler
                scaled = scaler.transform(values)
                seq_length = config.get('seq_lengths', {'15m': 15, '1h': 30, '4h': 45, '1d': 60}).get(timeframe, 30)
                X, _ = create_sequences(scaled, seq_length, [0])
                X = X.reshape((X.shape[0], seq_length, len(model_package.features)))
                pred = model_package.model.predict(X, verbose=0)
                pred_dummy = np.zeros((pred.shape[0], len(model_package.features)))
                pred_dummy[:, 0] = pred[:, 0]
                pred_rescaled = scaler.inverse_transform(pred_dummy)[:, 0]
                plt.plot(df.index[seq_length:], pred_rescaled, label=f"{model_type} 예측")
            else:
                pred = model_package.model.predict(n_periods=len(df)) if model_package.model.__class__.__name__ == 'ARIMA' else model_package.model.predict(df.reset_index()[['ds']])['yhat'].values
                plt.plot(df.index, pred, label=f"{model_type} 예측")
    
    plt.plot(df.index, df['close'], label="실제", linestyle='--')
    plt.title(f"{coin_name} - 실제 vs 예측 ({timeframe})")
    plt.xlabel("시간")
    plt.ylabel("가격")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, f"{coin_name}_actual_vs_pred_{timeframe}.png"))
    plt.close()

# process_model 전역 함수
def process_model(model_type, train_func, timeframe, coin_name):
    try:
        file_path = os.path.join(DATA_DIR, f"{coin_name}_{timeframe}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ 파일 없음: {file_path}")
            return timeframe, None

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"▶ {model_type} 학습 시작: {coin_name} ({timeframe})")
        result = train_func(df, timeframe)
        if result:
            backtest_result = backtest_predictions(result, df, timeframe)
            logger.info(f"백테스팅 결과 ({timeframe}): Sharpe Ratio = {backtest_result['sharpe_ratio']:.4f}")
        return timeframe, result
    except Exception as e:
        logger.error(f"모델 처리 실패 ({model_type}, {timeframe}): {e}")
        return timeframe, None

# 모델 학습
def train_models_grouped_by_type(coin_name, timeframes):
    today_str = datetime.today().strftime('%Y%m%d')

    models_dict = {'LSTM': {}, 'ARIMA': {}, 'Prophet': {}}
    with ProcessPoolExecutor(max_workers=config.get('max_workers', max(1, os.cpu_count() // 2))) as executor:
        futures = []
        for model_type, train_func in [('LSTM', train_lstm), ('ARIMA', train_arima), ('Prophet', train_prophet)]:
            for tf in tqdm(timeframes, desc=f"Training {model_type}"):
                futures.append(executor.submit(process_model, model_type, train_func, tf, coin_name))
        
        for future in futures:
            timeframe, result = future.result()
            if result:
                models_dict[model_type][timeframe] = result

    for model_type in models_dict:
        model_path = os.path.join(MODEL_DIR, f"{coin_name}_{today_str}_{model_type}.pkl")
        save_model_if_better(model_path, models_dict[model_type])

    for tf in timeframes:
        ensemble_pred = ensemble_predictions(models_dict, pd.read_csv(os.path.join(DATA_DIR, f"{coin_name}_{tf}.csv")).set_index('timestamp'), tf)
        logger.info(f"앙상블 예측 완료: {coin_name} ({tf})")
        
        visualize_results(models_dict, coin_name, pd.read_csv(os.path.join(DATA_DIR, f"{coin_name}_{tf}.csv")).set_index('timestamp'), tf)

if __name__ == "__main__":
    timeframes = config.get('timeframes', ['15m', '1h', '4h', '1d'])
    coins = config.get('coins', ['BTCUSDT'])
    for coin in coins:
        train_models_grouped_by_type(coin, timeframes)