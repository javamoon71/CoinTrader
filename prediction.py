import os
import re
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import threading
import queue
import warnings
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from matplotlib.dates import DateFormatter, AutoDateLocator
import matplotlib

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 설정
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

matplotlib.rc('axes', unicode_minus=False)
if sys.platform == 'win32':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='sans-serif')

# 설정값: 환경 변수 또는 상대 경로 사용
MODEL_DIR = os.getenv('MODEL_DIR')
DATA_DIR = os.getenv('DATA_DIR')
SAVE_DIR = os.getenv('SAVE_DIR')
os.makedirs(SAVE_DIR, exist_ok=True)

# model_package.py에서 ModelPackage 클래스 임포트
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.model_package import ModelPackage
    from utils.load_ohlcv import update_data
except ImportError:
    raise ImportError("필요한 모듈을 찾을 수 없습니다. 경로를 확인하세요.")

try:
    from prophet import Prophet
except ImportError:
    print("Prophet 라이브러리가 설치되지 않았습니다. Prophet 예측을 스킵합니다.")
    def predict_prophet(*args, **kwargs):
        return None, None, None

# ============================ 기술 지표 계산 함수 ============================
def calculate_all_indicators(df):
    df['close'] = df['close'].interpolate(method='linear')
    if 'volume' in df.columns:
        df['volume'] = df['volume'].interpolate(method='linear')

    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

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
        df['obv'] = 0

    return df.dropna().copy()

def find_latest_model(coin_name, model_type):
    pattern = re.compile(rf"{coin_name}USDT_(\d+)_({model_type})\.pkl")
    latest_date = ""
    latest_file = None
    for fname in os.listdir(MODEL_DIR):
        match = pattern.match(fname)
        if match:
            date_str = match.group(1)
            if date_str > latest_date:
                latest_date = date_str
                latest_file = fname
    return os.path.join(MODEL_DIR, latest_file) if latest_file else None

def load_model(symbol, model_type, timeframe):
    path = find_latest_model(symbol, model_type)
    if not path:
        return None
    try:
        model_dict = joblib.load(path)
        if timeframe not in model_dict:
            print(f"모델 파일에 '{timeframe}' 키가 없습니다.")
            return None
        model_package = model_dict[timeframe]
        
        if model_type == "LSTM":
            if not hasattr(model_package, 'mse_close') or not hasattr(model_package, 'mse_rsi'):
                print("LSTM 모델 패키지에 'mse_close' 또는 'mse_rsi' 속성이 없습니다.")
                return None
        elif model_type in ["ARIMA", "Prophet"]:
            if not hasattr(model_package, 'mse_close'):
                print(f"{model_type} 모델 패키지에 'mse_close' 속성이 없습니다.")
                return None
        
        if (datetime.now() - model_package.trained_at).days > 30:
            print(f"{model_type} 모델이 30일 이상 오래되었습니다. 재학습 추천.")
        return model_package
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None

def load_data(symbol, timeframe, days):
    try:
        df = update_data(symbol, timeframe, days)
        if df.empty:
            raise ValueError("데이터가 비었습니다.")
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"데이터 로딩 실패: {e}. 빈 데이터프레임 반환.")
        return pd.DataFrame()

freq_map = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440}

def calculate_steps(timeframe, days):
    total_minutes = days * 1440
    freq_minutes = freq_map.get(timeframe, 15)
    return total_minutes // freq_minutes

# ============================ predict_lstm 함수 (수정) ============================
def predict_lstm(symbol, timeframe, days, cancel_flag, lock):
    model_package = load_model(symbol, "LSTM", timeframe)
    if not model_package: 
        return None, None, None, None
    
    model = model_package.model
    scaler = model_package.scaler
    features = getattr(model_package, 'features', ['close', 'rsi']) 
    steps = calculate_steps(timeframe, days)
    
    df = load_data(symbol, timeframe, 60 + steps)
    if df.empty: 
        return None, None, None, None

    df_with_indicators = calculate_all_indicators(df)
    
    try:
        values = df_with_indicators[features].values
        scaled = scaler.transform(values)
    except ValueError as e:
        print(f"스케일링 오류: 학습된 피처와 예측 피처가 일치하지 않습니다. {e}")
        return None, None, None, None
        
    seq_length = 60
    last_sequence = scaled[-(seq_length+steps) : -steps].reshape((1, seq_length, len(features)))

    predictions_close = []
    predictions_rsi = []
    
    input_sequence = last_sequence
    
    for i in range(steps):
        with lock:
            if cancel_flag["cancel"]:
                return None, None, None, None
        
        pred_scaled = model.predict(input_sequence, verbose=0, batch_size=1)
        
        pred_close_scaled = pred_scaled[0, 0]
        predictions_close.append(pred_close_scaled)
        
        if 'rsi' in features:
            pred_rsi_scaled = pred_scaled[0, 1]
            predictions_rsi.append(pred_rsi_scaled)
        else:
            predictions_rsi.append(np.nan)

        new_input_scaled = np.zeros((1, 1, len(features)))
        new_input_scaled[0, 0, features.index('close')] = pred_close_scaled
        if 'rsi' in features:
            new_input_scaled[0, 0, features.index('rsi')] = pred_rsi_scaled
        
        for j, feature_name in enumerate(features):
            if feature_name not in ['close', 'rsi']:
                new_input_scaled[0, 0, j] = input_sequence[0, -1, j]

        input_sequence = np.concatenate((input_sequence[:, 1:, :], new_input_scaled), axis=1)

    forecast_scaled = np.zeros((len(predictions_close), len(features)))
    forecast_scaled[:, features.index('close')] = predictions_close
    if 'rsi' in features:
        forecast_scaled[:, features.index('rsi')] = predictions_rsi
        
    forecast_rescaled = scaler.inverse_transform(forecast_scaled)

    forecast_close = forecast_rescaled[:, features.index('close')]
    forecast_rsi = forecast_rescaled[:, features.index('rsi')]

    # ========================== RMSE 계산 로직 수정 ==========================
    rmse_close, rmse_rsi = None, None

    actual_values = df_with_indicators[features].iloc[-steps:].values
    actual_close_rescaled = actual_values[:, features.index('close')]
    
    if len(forecast_close) == len(actual_close_rescaled):
        rmse_close = np.sqrt(mean_squared_error(actual_close_rescaled, forecast_close))
    else:
        print(f"경고: RMSE 계산을 위한 실제값과 예측값의 길이가 다릅니다. 실제값: {len(actual_close_rescaled)}, 예측값: {len(forecast_close)}")

    if 'rsi' in features:
        actual_rsi_rescaled = actual_values[:, features.index('rsi')]
        if len(forecast_rsi) == len(actual_rsi_rescaled):
            rmse_rsi = np.sqrt(mean_squared_error(actual_rsi_rescaled, forecast_rsi))
        else:
            print(f"경고: RMSE 계산을 위한 실제값과 예측값의 길이가 다릅니다. 실제값: {len(actual_rsi_rescaled)}, 예측값: {len(forecast_rsi)}")

    return forecast_close, forecast_rsi, (model_package.mse_close, model_package.mse_rsi), (rmse_close, rmse_rsi)

def predict_arima(symbol, timeframe, days, cancel_flag, lock):
    model_package = load_model(symbol, "ARIMA", timeframe)
    if not model_package:
        return None, None, None
    steps = calculate_steps(timeframe, days)
    with lock:
        if cancel_flag["cancel"]:
            return None, None, None
    
    df = load_data(symbol, timeframe, 60 + days)
    if df.empty:
        return None, None, None
    
    series = df['close'].iloc[60:]
    
    forecast = model_package.model.predict(n_periods=steps)
    
    actual_values_to_compare = series.iloc[:steps]
    forecast_values_to_compare = forecast.iloc[:steps]
    rmse_close = np.sqrt(mean_squared_error(actual_values_to_compare, forecast_values_to_compare))

    return forecast, model_package.mse_close, rmse_close

def predict_prophet(symbol, timeframe, days, cancel_flag, lock):
    model_package = load_model(symbol, "Prophet", timeframe)
    if not model_package:
        return None, None, None
    steps = calculate_steps(timeframe, days)
    df = load_data(symbol, timeframe, 60 + days)
    if df.empty:
        return None, None, None
        
    df_prophet = df.reset_index()[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    future = model_package.model.make_future_dataframe(periods=steps, freq=f'{freq_map[timeframe]}min')
    
    with lock:
        if cancel_flag["cancel"]:
            return None, None, None
            
    df_forecast_compare = df_prophet.iloc[-steps:].copy()
    
    forecast = model_package.model.predict(future)
    forecast_values = forecast['yhat'][-steps:].values

    actual_values_to_compare = df_forecast_compare['y']
    forecast_values_to_compare = pd.Series(forecast_values, index=actual_values_to_compare.index)
    
    rmse_close = np.sqrt(mean_squared_error(actual_values_to_compare, forecast_values_to_compare))
    
    return forecast_values, model_package.mse_close, rmse_close

result_queue = queue.Queue()

def run_forecast_thread(cancel_flag, lock):
    timeframe = timeframe_var.get()
    days = int(period_var.get())
    symbol = symbol_var.get()
    
    with ThreadPoolExecutor() as executor:
        futures = {}
        
        result_queue.put("status:LSTM 예측 중...")
        futures["lstm"] = executor.submit(predict_lstm, symbol, timeframe, days, cancel_flag, lock)
        
        result_queue.put("status:ARIMA 예측 중...")
        futures["arima"] = executor.submit(predict_arima, symbol, timeframe, days, cancel_flag, lock)
        
        result_queue.put("status:Prophet 예측 중...")
        futures["prophet"] = executor.submit(predict_prophet, symbol, timeframe, days, cancel_flag, lock)
        
        results = {}
        for key, future in futures.items():
            result = future.result()
            with lock:
                if cancel_flag["cancel"]:
                    result_queue.put("cancelled")
                    return
            results[key] = result
            
    result_queue.put((results, timeframe, days, symbol))

processing = False

def process_results():
    global processing
    if processing:
        root.after(100, process_results)
        return
    processing = True
    try:
        result = result_queue.get_nowait()
        
        if isinstance(result, str) and result.startswith("status:"):
            status_label.config(text=result.split("status:")[1])
            processing = False
            root.after(100, process_results)
            return
            
        if result == "cancelled":
            status_label.config(text="❌ 예측이 취소되었습니다.")
            btn.config(state='normal')
            cancel_btn.config(state='disabled')
            return

        results, timeframe, days, symbol = result
        lstm_pred_close, lstm_pred_rsi, lstm_mse, lstm_rmse = results.get("lstm", (None, None, None, None))
        arima_pred, arima_mse, arima_rmse = results.get("arima", (None, None, None))
        prophet_pred, prophet_mse, prophet_rmse = results.get("prophet", (None, None, None))

        if all(pred is None for pred in [lstm_pred_close, arima_pred, prophet_pred]):
            status_label.config(text="❌ 모든 모델 예측 실패")
            btn.config(state='normal')
            cancel_btn.config(state='disabled')
            return

        steps_per_day_map = {'15m': 96, '1h': 24, '4h': 6, '1d': 1}
        steps_per_day = steps_per_day_map.get(timeframe, 96)
        total_steps = steps_per_day * days
        
        df_actual = load_data(symbol, timeframe, 60 + days)
        df_actual = calculate_all_indicators(df_actual)
        
        if df_actual.empty:
            status_label.config(text="❌ 데이터 로딩 실패")
            btn.config(state='normal')
            cancel_btn.config(state='disabled')
            return
            
        actual_timestamps = df_actual.index
        actual_prices = df_actual['close']
        actual_rsi = df_actual['rsi']
        
        start_time_pred = actual_timestamps[-1] + timedelta(minutes=freq_map[timeframe])
        timestamps_pred = [start_time_pred + timedelta(minutes=freq_map[timeframe] * i) for i in range(total_steps)]

        timestamps_all = actual_timestamps.tolist() + timestamps_pred
        
        for widget in frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(actual_timestamps, actual_prices, label='실제 시세', linestyle='-', color='black')
        
        if lstm_pred_close is not None:
            rmse_text = f", RMSE: {lstm_rmse[0]:.2f}" if lstm_rmse and lstm_rmse[0] is not None else ""
            ax1.plot(timestamps_pred, lstm_pred_close[:total_steps], label=f'LSTM 예측 (MSE: {lstm_mse[0]:.2f}{rmse_text})')
        if arima_pred is not None:
            rmse_text = f", RMSE: {arima_rmse:.2f}" if arima_rmse is not None else ""
            ax1.plot(timestamps_pred, arima_pred[:total_steps], label=f'ARIMA 예측 (MSE: {arima_mse:.2f}{rmse_text})')
        if prophet_pred is not None:
            rmse_text = f", RMSE: {prophet_rmse:.2f}" if prophet_rmse is not None else ""
            ax1.plot(timestamps_pred, prophet_pred[:total_steps], label=f'Prophet 예측 (MSE: {prophet_mse:.2f}{rmse_text})')
        
        ax1.set_title(f"{symbol}USDT {days}일 시세 예측 ({timeframe} 봉 기준)")
        ax1.set_ylabel("가격")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(actual_timestamps, actual_rsi, label='실제 RSI', linestyle='-', color='black')
        if lstm_pred_rsi is not None:
            rmse_text = f", RMSE: {lstm_rmse[1]:.2f}" if lstm_rmse and lstm_rmse[1] is not None else ""
            ax2.plot(timestamps_pred, lstm_pred_rsi[:total_steps], label=f'LSTM RSI 예측 (MSE: {lstm_mse[1]:.2f}{rmse_text})')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='과매수(70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='과매도(30)')
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("시간")
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 100)
        
        # x축 단위 변경 (시간봉에 따라)
        locator = AutoDateLocator(maxticks=10)
        
        if timeframe == '15m':
            formatter = DateFormatter('%H:%M')
        elif timeframe == '1h':
            formatter = DateFormatter('%m-%d %H')
        elif timeframe == '4h':
            formatter = DateFormatter('%m-%d')
        elif timeframe == '1d':
            formatter = DateFormatter('%Y-%m-%d')
        else:
            formatter = DateFormatter('%Y-%m-%d %H:%M')

        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        df_combined = pd.DataFrame(index=timestamps_all)
        df_combined['actual_close'] = actual_prices
        df_combined['actual_rsi'] = actual_rsi
        df_combined['lstm_close'] = pd.Series(lstm_pred_close, index=timestamps_pred)
        df_combined['lstm_rsi'] = pd.Series(lstm_pred_rsi, index=timestamps_pred)
        df_combined['arima_close'] = pd.Series(arima_pred, index=timestamps_pred)
        df_combined['prophet_close'] = pd.Series(prophet_pred, index=timestamps_pred)
        
        try:
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            df_combined.to_csv(os.path.join(SAVE_DIR, f"{symbol}USDT_{timeframe}_{days}d_forecast_{timestamp_str}.csv"))
            fig.savefig(os.path.join(SAVE_DIR, f"{symbol}USDT_{timeframe}_{days}d_forecast_{timestamp_str}.png"))
        except Exception as e:
            messagebox.showerror("저장 오류", f"결과 저장 실패: {e}")
        finally:
            plt.close(fig)

        status_label.config(text="✅ 예측 완료")
        btn.config(state='normal')
        cancel_btn.config(state='disabled')
    except queue.Empty:
        pass
    finally:
        processing = False
        root.after(100, process_results)

def show_forecast():
    with lock:
        cancel_flag["cancel"] = False
    status_label.config(text="⏳ 예측 시작 중...")
    btn.config(state='disabled')
    cancel_btn.config(state='normal')
    threading.Thread(target=run_forecast_thread, args=(cancel_flag, lock), daemon=True).start()
    root.after(100, process_results)

def cancel_forecast():
    with lock:
        cancel_flag["cancel"] = True
    status_label.config(text="⏳ 예측 취소 중...")
    cancel_btn.config(state='disabled')

# tkinter GUI
root = tk.Tk()
root.title("코인 시세 예측")
root.geometry("1000x800")

style = ttk.Style()
style.configure('Custom.TCombobox', font=('Helvetica', 10), padding=1)
style.configure('Custom.TButton', font=('Helvetica', 10), padding=1, background='#E0E0E0')
style.map('Custom.TButton', background=[('active', '#D0D0D0')])
style.configure('Custom.TLabel', font=('Helvetica', 10))

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=0)
root.grid_rowconfigure(2, weight=1)

top_frame = ttk.Frame(root)
top_frame.grid(row=0, column=0, pady=10, sticky=tk.EW)
top_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), weight=1)

ttk.Label(top_frame, text="코인 선택:", style='Custom.TLabel').grid(row=0, column=2, padx=0, sticky=tk.E)
symbol_var = tk.StringVar(value='BTC')
ttk.Combobox(top_frame, textvariable=symbol_var, values=['BTC', 'ETH', 'SOL'], state='readonly', width=12, style='Custom.TCombobox').grid(row=0, column=3, padx=0)

ttk.Label(top_frame, text="시간봉 선택:", style='Custom.TLabel').grid(row=0, column=4, padx=0, sticky=tk.E)
timeframe_var = tk.StringVar(value='15m')
ttk.Combobox(top_frame, textvariable=timeframe_var, values=['15m', '1h', '4h', '1d'], state='readonly', width=12, style='Custom.TCombobox').grid(row=0, column=5, padx=0)

ttk.Label(top_frame, text="예측 기간 (일):", style='Custom.TLabel').grid(row=0, column=6, padx=1, sticky=tk.E)
period_var = tk.StringVar(value='1')
ttk.Combobox(top_frame, textvariable=period_var, values=['1', '7', '15', '30'], state='readonly', width=12, style='Custom.TCombobox').grid(row=0, column=7, padx=0)

button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, pady=10, sticky=tk.EW)
button_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

status_label = ttk.Label(button_frame, text="🟢 대기 중", style='Custom.TLabel')
status_label.grid(row=0, column=2, padx=0, sticky=tk.W)

btn = ttk.Button(button_frame, text="예측 실행", command=show_forecast, style='Custom.TButton')
btn.grid(row=0, column=3, padx=0)

cancel_btn = ttk.Button(button_frame, text="예측 취소", command=cancel_forecast, state='disabled', style='Custom.TButton')
cancel_btn.grid(row=0, column=4, padx=0)

frame = ttk.Frame(root)
frame.grid(row=2, column=0, sticky=tk.NSEW)

cancel_flag = {"cancel": False}
lock = threading.Lock()

root.after(100, process_results)
root.mainloop()

