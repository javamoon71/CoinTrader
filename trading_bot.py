import os
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
import warnings
import sys
import re
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

# --- 경로 설정: PyInstaller EXE 파일 환경에 맞게 수정 ---
if getattr(sys, 'frozen', False):
    # PyInstaller로 패키징된 경우, sys._MEIPASS는 임시 압축 해제 경로를 가리킵니다.
    application_path = sys._MEIPASS
else:
    # 일반적인 Python 환경인 경우, 현재 스크립트의 디렉토리를 사용합니다.
    application_path = os.path.dirname(os.path.abspath(__file__))

# .env 파일 로드 (exe와 같은 디렉토리에 있다고 가정)
dotenv_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(dotenv_path):
    print(f"Error: .env file not found at {dotenv_path}. Please place it in the same directory as the executable.")
    sys.exit(1)
load_dotenv(dotenv_path)

import dateparser
print(os.path.join(os.path.dirname(dateparser.__file__), 'data'))

# utils 폴더의 모듈들을 임포트 (PyInstaller가 임시 경로에 압축을 풀면 해당 폴더가 루트에 위치하게 됨)
try:
    from utils.telegram import send_telegram_channel_message
    from utils.model_package import ModelPackage
    from utils.load_ohlcv import update_data
except ImportError as e:
    print(f"Error importing required modules from utils: {e}")
    # 함수/클래스 더미 선언
    def send_telegram_channel_message(*args, **kwargs):
        pass
    def update_data(*args, **kwargs):
        raise NotImplementedError("update_data 함수를 로드할 수 없습니다.")
    class ModelPackage:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ModelPackage 클래스를 로드할 수 없습니다.")
    sys.exit(1)

try:
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY or BINANCE_API_SECRET not found in .env file.")
    client = Client(api_key, api_secret)
except Exception as e:
    print(f"바이낸스 API 클라이언트 초기화 실패: {e}")
    print("API 키를 확인하거나 .env 파일 설정을 검토하세요. 프로그램이 종료됩니다.")
    sys.exit(1)


# ====================================================================
# [변경 사항] - 골든 크로스 및 데드 크로스 지표 추가
# ====================================================================
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
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
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

    # 골든 크로스 및 데드 크로스 지표 추가
    # 골든 크로스: 단기 이평선(ma25)이 장기 이평선(ma99)을 상향 돌파
    df['golden_cross'] = (df['ma25'].shift(1) <= df['ma99'].shift(1)) & (df['ma25'] > df['ma99'])
    # 데드 크로스: 단기 이평선(ma25)이 장기 이평선(ma99)을 하향 돌파
    df['dead_cross'] = (df['ma25'].shift(1) >= df['ma99'].shift(1)) & (df['ma25'] < df['ma99'])

    return df.dropna().copy()
# ====================================================================


def find_latest_model(coin_name, model_type, model_dir):
    pattern = re.compile(rf"{coin_name}USDT_(\d+)_({model_type})\.pkl")
    latest_date = ""
    latest_file = None
    if not os.path.exists(model_dir):
        return None
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            date_str = match.group(1)
            if date_str > latest_date:
                latest_date = date_str
                latest_file = fname
    return os.path.join(model_dir, latest_file) if latest_file else None


def load_model(symbol, model_type, model_dir):
    path = find_latest_model(symbol, model_type, model_dir)
    if not path:
        return None
    try:
        model_dict = joblib.load(path)
        return model_dict
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None


def load_data(symbol, timeframe, days, data_dir):
    try:
        df = update_data(symbol, timeframe, days, data_dir=data_dir)
        if df.empty:
            raise ValueError("데이터가 비었습니다.")
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"데이터 로딩 실패: {e}. 빈 데이터프레임 반환.")
        return pd.DataFrame()


class TradingBot:
    def __init__(self, symbol, log_widget_ref, profit_label_ref, return_label_ref, buy_count_label_ref, sell_count_label_ref, model_dir, data_dir):
        self.symbol = symbol
        self.log_widget_ref = log_widget_ref
        self.profit_label_ref = profit_label_ref
        self.return_label_ref = return_label_ref
        self.buy_count_label_ref = buy_count_label_ref
        self.sell_count_label_ref = sell_count_label_ref
        
        self.position = None
        self.buy_price = 0
        self.trade_amount = 10 
        self.is_running = False

        self.buy_threshold_strong = 3
        self.buy_threshold_medium = 2
        
        self.sell_threshold_strong = -3
        
        self.profit_take_ratio = 5
        self.stop_loss_ratio = -5

        self.model_packages = {}
        self.supported_timeframes = ['15m', '1h', '4h', '1d']
        
        self.trade_history = []
        self.total_profit = 0
        self.initial_capital = 1000
        self.current_capital = self.initial_capital
        
        self.buy_count = 0
        self.sell_count = 0
        
        self.MODEL_DIR = model_dir
        self.DATA_DIR = data_dir

        self._load_and_validate_models()

    def _load_and_validate_models(self):
        self.log_message(f"[{self.symbol}] 여러 타임프레임 모델 로딩 시도...")
        model_dict = load_model(self.symbol, 'LSTM', self.MODEL_DIR)
        
        if not model_dict:
            self.log_message("⚠️ 모델 로딩에 실패했습니다. 봇이 시작되지 않습니다.")
            self.model_packages = None
            return

        for tf in self.supported_timeframes:
            if tf in model_dict:
                self.model_packages[tf] = model_dict[tf]
                self.log_message(f"✅ {tf} 모델 로드 성공.")
            else:
                self.log_message(f"❌ {tf} 모델을 찾을 수 없습니다. 이 모델은 예측에 사용되지 않습니다.")
        
        if not self.model_packages:
            self.log_message("⚠️ 유효한 모델이 없어 봇이 시작되지 않습니다.")
            self.model_packages = None


    def log_message(self, message):
        self.log_widget_ref.after(0, self._insert_log, message)

    def _insert_log(self, message):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_message = f"[{timestamp}] {message}\n"
            self.log_widget_ref.configure(state='normal')
            
            # [변경] 최신 메시지를 가장 위(1.0)에 삽입합니다.
            self.log_widget_ref.insert('1.0', full_message)
            
            self.log_widget_ref.configure(state='disabled')
            app.update_status()

    def get_latest_price(self):
        try:
            ticker = client.get_ticker(symbol=f'{self.symbol}USDT')
            return float(ticker['lastPrice'])
        except Exception as e:
            self.log_message(f"❌ Error fetching current price: {e}")
            return None

    def execute_buy_order(self, price, amount):
        self.log_message(f"✅ 매수 주문 실행: {self.symbol} {amount:.4f}개, 가격: {price:.2f} USDT (가상)")
        
        self.position = 'long'
        self.buy_price = price
        
        self.trade_history.append({
            "type": "BUY",
            "price": price,
            "quantity": amount,
            "timestamp": datetime.now()
        })
        self.buy_count += 1
        self.log_message(f"🟢 매수 성공. 매수 가격: {self.buy_price:.2f}")

    def execute_sell_order(self, price, amount):
        self.log_message(f"🔴 매도 주문 실행: {self.symbol} {amount:.4f}개, 가격: {price:.2f} USDT (가상)")
        
        self.trade_history.append({
            "type": "SELL",
            "price": price,
            "quantity": amount,
            "timestamp": datetime.now()
        })
        
        profit = (price - self.buy_price) * amount
        self.total_profit += profit
        self.current_capital += profit
        
        self.log_message(f"🔵 매도 성공. 거래 수익: {profit:.2f} USDT. 누적 수익: {self.total_profit:.2f} USDT")
        
        self.position = None
        self.buy_price = 0
        
        self.sell_count += 1
        
        self.update_profit_labels()
        
    def update_profit_labels(self):
        self.profit_label_ref.after(0, lambda: self.profit_label_ref.config(text=f"누적 수익: {self.total_profit:.2f} USDT"))
        
        if self.initial_capital > 0:
            return_rate = (self.current_capital - self.initial_capital) / self.initial_capital * 100
            self.return_label_ref.after(0, lambda: self.return_label_ref.config(text=f"총 수익률: {return_rate:.2f}%"))
        else:
            self.return_label_ref.after(0, lambda: self.return_label_ref.config(text=f"총 수익률: 계산 불가"))

        self.buy_count_label_ref.after(0, lambda: self.buy_count_label_ref.config(text=f"매수 횟수: {self.buy_count}회"))
        self.sell_count_label_ref.after(0, lambda: self.sell_count_label_ref.config(text=f"매도 횟수: {self.sell_count}회"))


    # ====================================================================
    # [변경 사항] - 전략 로직에 골든/데드 크로스 신호 추가
    # ====================================================================
    def run_trading_strategy(self):
        if not self.model_packages:
            self.log_message("⚠️ 모델이 로드되지 않아 전략을 실행할 수 없습니다.")
            return

        predicted_prices = []
        current_price = self.get_latest_price()
        if current_price is None:
            return

        for tf, model_package in self.model_packages.items():
            try:
                days_to_load = {'15m': 1, '1h': 7, '4h': 30, '1d': 365}.get(tf, 1)
                df_raw = load_data(self.symbol, tf, days_to_load, self.DATA_DIR)
                
                if df_raw.empty or len(df_raw) < 60:
                    self.log_message(f"❌ {tf} 데이터 로딩 실패 또는 부족. 예측에서 제외.")
                    continue
                
                df_with_indicators = calculate_all_indicators(df_raw.copy())
                
                features = model_package.features
                scaled_data = model_package.scaler.transform(df_with_indicators[features].values)
                last_sequence = scaled_data[-60:].reshape((1, 60, len(features)))
                
                pred_scaled = model_package.model.predict(last_sequence, verbose=0)
                
                forecast_dummy = np.zeros((1, len(features)))
                forecast_dummy[0, features.index('close')] = pred_scaled[0, 0]
                
                forecast_rescaled = model_package.scaler.inverse_transform(forecast_dummy)
                predicted_future_price = forecast_rescaled[0, features.index('close')]
                
                predicted_prices.append(predicted_future_price)
                self.log_message(f"✅ {tf} 예측 가격: {predicted_future_price:.2f} USDT")

            except Exception as e:
                self.log_message(f"❌ {tf} 예측 과정에서 오류 발생: {e}. 예측에서 제외.")
                continue

        if not predicted_prices:
            self.log_message("⚠️ 모든 타임프레임 예측에 실패했습니다. 다음 주기에 재시도.")
            return

        final_predicted_price = np.mean(predicted_prices)
        price_diff_ratio = (final_predicted_price - current_price) / current_price * 100

        # 최신 지표 신호 확인
        latest_indicators = df_with_indicators.iloc[-1]
        golden_cross_signal = latest_indicators['golden_cross']
        dead_cross_signal = latest_indicators['dead_cross']
        
        self.log_message(f"📝 종합 예측 정보: 현재 가격 {current_price:.2f}, 최종 예측 가격 {final_predicted_price:.2f} (변동률: {price_diff_ratio:.2f}%)")
        
        if self.position is None:
            # 매수 신호: 모델 예측(강력) OR 골든 크로스 발생 시
            if (price_diff_ratio > self.buy_threshold_strong) or golden_cross_signal:
                if golden_cross_signal:
                    self.log_message("⭐ 골든 크로스 신호 포착!")
                    send_telegram_channel_message(self.symbol, "골든 크로스 신호 포착! 매수 고려", level="🚨🚨")
                else:
                    self.log_message("⭐ 적극 매수 신호 포착!")
                    send_telegram_channel_message(self.symbol, f"강력 매수 신호 포착! 예측 변동률: {price_diff_ratio:.2f}%", level="🚨🚨🚨")

                trade_amount_coin = self.trade_amount / current_price
                self.execute_buy_order(current_price, trade_amount_coin)
            elif price_diff_ratio > self.buy_threshold_medium:
                self.log_message("✅ 매수 신호 포착!")
                trade_amount_coin = self.trade_amount / current_price
                self.execute_buy_order(current_price, trade_amount_coin)
            else:
                self.log_message("🟢 관망 중 (매수)")
        
        elif self.position == 'long':
            profit_ratio = (current_price - self.buy_price) / self.buy_price * 100
            
            # 매도 신호: 이익 실현, 손절, 모델 예측(강력), 데드 크로스 중 하나라도 해당될 시
            if (profit_ratio >= self.profit_take_ratio) or \
               (profit_ratio <= self.stop_loss_ratio) or \
               (price_diff_ratio < self.sell_threshold_strong) or \
               dead_cross_signal:

                if profit_ratio >= self.profit_take_ratio:
                    self.log_message(f"📈 이익 실현 (수익률: +{profit_ratio:.2f}%)")
                elif profit_ratio <= self.stop_loss_ratio:
                    self.log_message(f"📉 손절 (손실률: {profit_ratio:.2f}%)")
                elif price_diff_ratio < self.sell_threshold_strong:
                    self.log_message("🚨 적극 매도 신호 포착! (보유 중)")
                    send_telegram_channel_message(self.symbol, "강력 매도 신호 포착! 보유 자산 손절/이익 실현 고려", level="🚨🚨🚨")
                elif dead_cross_signal:
                    self.log_message("🚨 데드 크로스 신호 포착! (보유 중)")
                    send_telegram_channel_message(self.symbol, "데드 크로스 신호 포착! 보유 자산 매도 고려", level="🚨🚨")

                trade_amount_coin = self.trade_amount / self.buy_price
                self.execute_sell_order(current_price, trade_amount_coin)
            else:
                self.log_message(f"🟢 보유 중 (수익률: {profit_ratio:.2f}%)")
# ====================================================================


class TradingBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("자동 트레이딩 봇 v1.0")
        self.root.geometry("800x600")

        self.bot = None
        self.run_thread = None
        self.symbol = tk.StringVar(value='BTC')
        
        # models, data 경로를 관리하는 변수 추가
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        self.model_dir = tk.StringVar(value=os.path.join(exe_dir, 'models'))
        self.data_dir = tk.StringVar(value=os.path.join(exe_dir, 'data'))
        
        # 필요한 디렉토리 생성
        os.makedirs(self.model_dir.get(), exist_ok=True)
        os.makedirs(self.data_dir.get(), exist_ok=True)
        
        self.create_widgets()
        self.update_status()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main_frame, text="봇 설정", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        path_frame = ttk.LabelFrame(config_frame, text="경로 설정", padding="10")
        path_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(path_frame, text="모델 폴더:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_path_entry = ttk.Entry(path_frame, textvariable=self.model_dir, width=50)
        self.model_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(path_frame, text="선택", command=self.select_model_dir).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(path_frame, text="데이터 폴더:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.data_path_entry = ttk.Entry(path_frame, textvariable=self.data_dir, width=50)
        self.data_path_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(path_frame, text="선택", command=self.select_data_dir).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(config_frame, text="코인 선택:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_combobox = ttk.Combobox(config_frame, textvariable=self.symbol, 
                                            values=['BTC', 'ETH', 'SOL', 'XRP'], state='readonly', width=10)
        self.symbol_combobox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.symbol_combobox.bind("<<ComboboxSelected>>", self.reinitialize_bot)

        status_frame = ttk.LabelFrame(main_frame, text="봇 상태", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label_text = tk.StringVar(value="봇 상태: 정지")
        self.price_label_text = tk.StringVar(value="현재 시세: -")
        self.position_label_text = tk.StringVar(value="포지션: 없음")
        self.buy_price_label_text = tk.StringVar(value="매수 가격: -")
        
        self.total_profit_text = tk.StringVar(value="누적 수익: 0.00 USDT")
        self.total_return_text = tk.StringVar(value="총 수익률: 0.00%")
        
        self.buy_count_text = tk.StringVar(value="매수 횟수: 0회")
        self.sell_count_text = tk.StringVar(value="매도 횟수: 0회")


        ttk.Label(status_frame, textvariable=self.status_label_text, font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.price_label_text, font=('Helvetica', 10)).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.position_label_text, font=('Helvetica', 10)).grid(row=0, column=1, sticky=tk.W, padx=20)
        ttk.Label(status_frame, textvariable=self.buy_price_label_text, font=('Helvetica', 10)).grid(row=1, column=1, sticky=tk.W, padx=20)
        
        ttk.Label(status_frame, textvariable=self.total_profit_text, font=('Helvetica', 10)).grid(row=0, column=2, sticky=tk.W, padx=20)
        ttk.Label(status_frame, textvariable=self.total_return_text, font=('Helvetica', 10)).grid(row=1, column=2, sticky=tk.W, padx=20)
        
        ttk.Label(status_frame, textvariable=self.buy_count_text, font=('Helvetica', 10)).grid(row=0, column=3, sticky=tk.W, padx=20)
        ttk.Label(status_frame, textvariable=self.sell_count_text, font=('Helvetica', 10)).grid(row=1, column=3, sticky=tk.W, padx=20)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(control_frame, text="봇 시작", command=self.start_bot)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="봇 정지", command=self.stop_bot, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="거래 로그", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled', height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.bot = TradingBot(self.symbol.get(), self.log_text, self.total_profit_text, self.total_return_text, self.buy_count_text, self.sell_count_text, self.model_dir.get(), self.data_dir.get())
        if not self.bot.model_packages:
            self.start_btn.config(state=tk.DISABLED)
            
    def select_model_dir(self):
        new_path = filedialog.askdirectory(initialdir=self.model_dir.get())
        if new_path:
            self.model_dir.set(new_path)
            self.reinitialize_bot()

    def select_data_dir(self):
        new_path = filedialog.askdirectory(initialdir=self.data_dir.get())
        if new_path:
            self.data_dir.set(new_path)
            self.reinitialize_bot()

    def reinitialize_bot(self, event=None):
        if self.bot.is_running:
            messagebox.showwarning("봇 실행 중", "봇을 정지한 후 설정을 변경해주세요.")
            if isinstance(event, tk.Event):
                self.symbol_combobox.set(self.bot.symbol)
            return

        selected_symbol = self.symbol.get()
        
        if not os.path.isdir(self.model_dir.get()):
            messagebox.showerror("오류", f"모델 폴더를 찾을 수 없습니다: {self.model_dir.get()}")
            return
        if not os.path.isdir(self.data_dir.get()):
            messagebox.showerror("오류", f"데이터 폴더를 찾을 수 없습니다: {self.data_dir.get()}")
            return
            
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
        self.bot = TradingBot(selected_symbol, self.log_text, self.total_profit_text, self.total_return_text, self.buy_count_text, self.sell_count_text, self.model_dir.get(), self.data_dir.get())
        self.update_status()
        self.total_profit_text.set("누적 수익: 0.00 USDT")
        self.total_return_text.set("총 수익률: 0.00%")
        self.buy_count_text.set("매수 횟수: 0회")
        self.sell_count_text.set("매도 횟수: 0회")

        if not self.bot.model_packages:
            self.start_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
        self.bot.log_message(f"코인을 {selected_symbol}으로 변경했습니다. 모델 및 데이터 경로를 다시 로드했습니다.")


    def start_bot(self):
        if self.bot and self.bot.model_packages and not self.bot.is_running:
            self.bot.is_running = True
            self.bot.log_message(f"{self.bot.symbol} 봇을 시작합니다.")
            self.run_thread = threading.Thread(target=self.run_bot_loop, daemon=True)
            self.run_thread.start()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_status_loop()
        elif not self.bot.model_packages:
            messagebox.showerror("오류", "모델 로딩에 실패하여 봇을 시작할 수 없습니다.")
            self.start_btn.config(state=tk.DISABLED)

    def stop_bot(self):
        if self.bot and self.bot.is_running:
            self.bot.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.bot.log_message("봇을 정지합니다.")

    def run_bot_loop(self):
        while self.bot.is_running:
            self.bot.run_trading_strategy()
            time.sleep(60)

    def update_status(self):
        if self.bot:
            current_price = self.bot.get_latest_price()
            if current_price:
                self.price_label_text.set(f"현재 시세: {current_price:.2f} USDT")
            
            if self.bot.position == 'long':
                self.position_label_text.set(f"포지션: 보유 중 ({self.bot.symbol}USDT)")
                self.buy_price_label_text.set(f"매수 가격: {self.bot.buy_price:.2f} USDT")
            else:
                self.position_label_text.set(f"포지션: 없음")
                self.buy_price_label_text.set(f"매수 가격: -")
                
            self.status_label_text.set(f"봇 상태: {'🟢 실행 중' if self.bot.is_running else '🔴 정지'}")

    def update_status_loop(self):
        self.update_status()
        if self.bot.is_running:
            self.root.after(5000, self.update_status_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotApp(root)
    root.mainloop()