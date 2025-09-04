import os
import sys
import pandas as pd
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import time
import glob
import matplotlib.font_manager as fm

# Matplotlib 한글 폰트 설정
# 시스템에 설치된 한글 폰트 경로를 찾아서 설정합니다.
# 일반적으로 'Malgun Gothic' 또는 'NanumGothic'이 사용됩니다.
try:
    font_path = fm.findfont(fm.FontProperties(family='Malgun Gothic'))
    plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False) # 한글 폰트 사용 시 - 부호 깨짐 방지
    print("✅ Matplotlib 한글 폰트 설정 완료: Malgun Gothic")
except Exception:
    print("⚠️ Malgun Gothic 폰트를 찾을 수 없습니다. 다른 한글 폰트(예: NanumGothic)를 시도하거나 설치해주세요.")

# `utils` 폴더 경로 추가 (상위 폴더에 위치한 경우)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ⚠️ 환경 변수 확인 및 설정
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 프로젝트의 root 디렉토리 (부모 디렉토리)
root_dir = os.path.dirname(current_file_path)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# ✅ Binance API 키 설정
client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

DATA_DIR = os.path.join(root_dir, 'CoinTrader\\' + os.getenv('DATA_DIR', 'data'))
MODEL_DIR = os.path.join(root_dir, 'CoinTrader\\' + os.getenv('MODEL_DIR', 'models'))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================ 바이낸스 데이터 로드 함수 ============================
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

def update_data(symbol, interval, days, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    full_symbol = symbol.upper() + 'USDT'
    filename = os.path.join(data_dir, f"{full_symbol}_{interval}.csv")
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
    print(f"✅ {full_symbol} 데이터 저장 완료: {filename}")
    print(f"🧮 총 데이터 수: {len(combined_df)}행")
    return combined_df

# ============================ 기술 지표 계산 함수 ============================
def calculate_all_indicators(df):
    df['close'] = df['close'].interpolate(method='linear')
    df['volume'] = df['volume'].interpolate(method='linear')
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss.where(loss != 0, 1e-10)
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
        direction = np.sign(close.diff().fillna(0))
        obv = (volume * direction).cumsum()
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
    df['golden_cross'] = ((df['ma25'].shift(1) <= df['ma99'].shift(1)) & (df['ma25'] > df['ma99'])).astype(int)
    df['dead_cross'] = ((df['ma25'].shift(1) >= df['ma99'].shift(1)) & (df['ma25'] < df['ma99'])).astype(int)
    return df.dropna().copy()

# ============================ GUI 애플리케이션 클래스 ============================
class BacktestApp(tk.Tk):
    def __init__(self, data_df, model_path):
        super().__init__()
        self.title("코인 백테스트 시뮬레이션")
        self.geometry("1400x800") # ✨ GUI 창 가로 크기 확대
        
        self.data_df = data_df
        self.model_path = model_path
        
        # 모델 로드
        try:
            models_dict = joblib.load(self.model_path)
            self.model_package = models_dict.get('1h')
            
            if self.model_package is None:
                print("❌ '1h' 모델이 딕셔너리에 없습니다.")
                self.model = None
            else:
                self.model = self.model_package.model
                self.scaler = self.model_package.scaler
                self.features = self.model_package.features
                print("✅ 모델 로드 성공: '1h' 모델을 선택했습니다.")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model = None

        self.initial_cash = 1000
        self.cash = self.initial_cash
        self.position = 0.0
        self.trade_log = []
        
        # 백테스트 시작 인덱스를 초기 데이터의 60번째로 설정합니다.
        # 시뮬레이션이 진행됨에 따라 이 인덱스가 증가합니다.
        self.current_index = 60
        self.running = False
        
        # 데이터 전처리
        self.data_with_indicators = calculate_all_indicators(self.data_df)
        self.data_with_indicators['timestamp_str'] = self.data_with_indicators['timestamp'].dt.strftime('%m-%d %H:%M')
        self.seq_length = 60
        
        # GUI 위젯 설정
        self.setup_ui()
        
    def setup_ui(self):
        # 상단 컨트롤 패널
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="백테스트 시작", command=self.start_backtest)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="중지", command=self.stop_backtest, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="준비됨")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 차트 영역
        self.fig, self.ax = plt.subplots(figsize=(14, 6)) # ✨ 그래프 가로 크기 확대
        self.ax.set_title("BTCUSDT 가격 시뮬레이션")
        self.ax.set_xlabel("시간")
        self.ax.set_ylabel("가격")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)
        
        # 매수/매도 로그를 표시할 영역
        log_frame = ttk.Frame(self)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 로그 텍스트 위젯과 스크롤바
        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", font=("Malgun Gothic", 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
    def start_backtest(self):
        if not self.model:
            self.status_label.config(text="오류: 모델이 로드되지 않았습니다.")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="백테스트 진행 중...")
        
        # 차트 초기화
        self.ax.clear()
        self.ax.set_title("BTCUSDT 가격 시뮬레이션")
        self.ax.set_xlabel("시간")
        self.ax.set_ylabel("가격")
        self.ax.grid(True)
        self.canvas.draw()

        # 로그 창 초기화
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        self.trade_log = []
        
        # 백테스트 루프 시작
        self.after(1, self.run_tick)

    def stop_backtest(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="백테스트 중지됨")
        self.show_results()

    def update_log_display(self, log_entry):
        """매수/매도 로그를 GUI 텍스트 위젯에 추가합니다."""
        action = log_entry['action']
        price = log_entry['price']
        timestamp = log_entry['time'].strftime('%Y-%m-%d %H:%M')
        
        log_message = f"[{timestamp}] {action} - Price: {price:.2f}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, log_message)
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END) # 스크롤을 맨 아래로 이동

    def run_tick(self):
        if not self.running or self.current_index >= len(self.data_with_indicators) - 1:
            self.stop_backtest()
            return

        # 현재 데이터 가져오기
        current_data = self.data_with_indicators.iloc[self.current_index - self.seq_length : self.current_index]
        current_tick = self.data_with_indicators.iloc[self.current_index]
        
        # 예측 (기존 로직 유지)
        values = current_data[self.features].values
        scaled_values = self.scaler.transform(values)
        X = scaled_values.reshape(1, self.seq_length, len(self.features))
        
        predicted_scaled = self.model.predict(X, verbose=0)
        
        pred_dummy = np.zeros((1, len(self.features)))
        features_to_predict_indices = [self.features.index(f) for f in ['close', 'rsi']]
        pred_dummy[:, features_to_predict_indices] = predicted_scaled
        predicted_values = self.scaler.inverse_transform(pred_dummy)[0]
        
        predicted_next_close = predicted_values[features_to_predict_indices[0]]
        current_close = current_tick['close']
        
        action = None
        
        # === 거래 전략 적용: MA99를 기준으로 매수/매도 필터링 ===
        
        # 매수 조건: 포지션이 없고, 골든 크로스가 발생했으며, 가격이 MA99 아래에 있을 때
        # MA99 아래에서만 매수하여 추세선 돌파를 노리는 전략입니다.
        if (self.position == 0 and 
            current_tick['golden_cross'] == 1 and 
            current_close < current_tick['ma99']):
            buy_price = current_close
            self.position = self.cash / buy_price
            self.cash = 0
            self.trade_log.append({'action': 'BUY (Golden Cross & Below MA99)', 'price': buy_price, 'time': current_tick['timestamp']})
            self.update_log_display(self.trade_log[-1])
            action = 'BUY'

        # 매도 조건: 포지션이 있고, 데드 크로스가 발생했으며, 가격이 MA99 위에 있을 때
        # MA99 위에서만 매도하여 추세선 이탈 시 수익을 확정하는 전략입니다.
        elif (self.position > 0 and 
              current_tick['dead_cross'] == 1 and 
              current_close > current_tick['ma99']):
            sell_price = current_close
            self.cash = self.position * sell_price
            self.position = 0
            self.trade_log.append({'action': 'SELL (Dead Cross & Above MA99)', 'price': sell_price, 'time': current_tick['timestamp']})
            self.update_log_display(self.trade_log[-1])
            action = 'SELL'
            
        # 손절매 조건: 포지션이 있고, 1시간 봉에서 3% 이상 급락이 발생했을 때
        # 이 조건은 가격 위치와 관계없이 리스크 관리를 위해 항상 적용됩니다.
        elif self.position > 0 and (current_tick['open'] > 0) and ((current_tick['open'] - current_tick['close']) / current_tick['open'] > 0.03):
            sell_price = current_close
            self.cash = self.position * sell_price
            self.position = 0
            self.trade_log.append({'action': 'SELL (Sharp Drop)', 'price': sell_price, 'time': current_tick['timestamp']})
            self.update_log_display(self.trade_log[-1])
            action = 'SELL'
        
        # === 한 달 데이터만 보여주는 차트 업데이트 로직 ===
        # 한 달을 30일(720시간)로 가정
        window_size_hours = 30 * 24
        start_index = max(0, self.current_index - window_size_hours)
        end_index = self.current_index + 1
        
        # 현재 윈도우 데이터 슬라이싱
        window_df = self.data_with_indicators.iloc[start_index:end_index].copy()
        
        # 기존 차트 내용 지우기
        self.ax.clear()
        
        # 차트 기본 설정 다시 적용
        self.ax.set_title("BTCUSDT 가격 시뮬레이션")
        self.ax.set_xlabel("시간")
        self.ax.set_ylabel("가격")
        self.ax.grid(True)
        
        # 현재 윈도우의 시세와 추세선 그리기
        self.ax.plot(window_df['timestamp'], window_df['close'], 'k-', label='Price')
        self.ax.plot(window_df['timestamp'], window_df['ma99'], 'b-', label='MA99')
        self.ax.plot(window_df['timestamp'], window_df['ma25'], 'y-', label='MA25')
        
        # 윈도우 내의 매수/매도 지점 찾아서 그리기
        buy_logs_in_window = [log for log in self.trade_log if window_df['timestamp'].iloc[0] <= log['time'] <= window_df['timestamp'].iloc[-1]]
        
        buy_times = [log['time'] for log in buy_logs_in_window if log['action'].startswith('BUY')]
        sell_times = [log['time'] for log in buy_logs_in_window if log['action'].startswith('SELL')]
        
        buy_prices = [self.data_with_indicators[self.data_with_indicators['timestamp'] == t]['close'].iloc[0] for t in buy_times]
        sell_prices = [self.data_with_indicators[self.data_with_indicators['timestamp'] == t]['close'].iloc[0] for t in sell_times]

        self.ax.plot(buy_times, buy_prices, 'go', markersize=8, label='Buy')
        self.ax.plot(sell_times, sell_prices, 'ro', markersize=8, label='Sell')

        # x축 포맷 설정
        self.fig.autofmt_xdate(rotation=45)

        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
        self.current_index += 1
        
        # 10ms 후 다음 틱 스케줄링
        self.after(10, self.run_tick)

    def show_results(self):
        final_value = self.cash + self.position * self.data_with_indicators['close'].iloc[-1]
        total_return = (final_value / self.initial_cash - 1) * 100
        
        # 매수, 매도 횟수 계산
        buy_count = len([log for log in self.trade_log if log['action'].startswith('BUY')])
        sell_count = len([log for log in self.trade_log if log['action'].startswith('SELL')])

        result_text = f"💰 초기 자산: ${self.initial_cash:.2f}\n"
        result_text += f"💸 최종 자산: ${final_value:.2f}\n"
        result_text += f"📈 총 수익률: {total_return:.2f}%\n"
        result_text += f"📊 총 매수 횟수: {buy_count}회\n"
        result_text += f"📊 총 매도 횟수: {sell_count}회"

        self.status_label.config(text=result_text)

if __name__ == "__main__":
    # 데이터 로드: CSV 파일에서 기존 데이터 로드
    data_file = os.path.join(DATA_DIR, "BTCUSDT_1h.csv")
    if not os.path.exists(data_file):
        print("❌ 'BTCUSDT_1h.csv' 데이터 파일이 없습니다. `training.py`를 먼저 실행하여 데이터를 생성하세요.")
        sys.exit(1)
        
    btc_df = pd.read_csv(data_file, parse_dates=['timestamp'])
    print(f"✅ 'BTCUSDT_1h.csv' 데이터 로드 완료. 총 {len(btc_df)}행")

    # 모델 경로 설정: 'models' 폴더에서 가장 최근의 LSTM 모델 파일을 찾습니다.
    model_files = glob.glob(os.path.join(MODEL_DIR, "BTCUSDT_*_LSTM.pkl"))
    if not model_files:
        print("❌ 'models' 폴더에 LSTM 모델 파일이 없습니다. `training.py`를 먼저 실행하여 모델을 생성하세요.")
        sys.exit(1)
    
    # 가장 최근에 수정된 파일 찾기
    model_file = max(model_files, key=os.path.getmtime)
    print(f"✅ 가장 최근에 생성된 모델 파일 로드: {os.path.basename(model_file)}")
    
    # GUI 애플리케이션 실행
    app = BacktestApp(btc_df, model_file)
    app.mainloop()
