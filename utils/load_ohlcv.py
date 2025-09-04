# 바이낸스에서 OHLCV 데이터를 로드하고 저장하는 스크립트

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv

current_file_path = os.path.dirname(os.path.abspath(__file__))
# 프로젝트의 root 디렉토리 (부모 디렉토리)
root_dir = os.path.dirname(current_file_path)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

# ✅ Binance API 키 설정
client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

# 설정값: 환경 변수 또는 상대 경로 사용
DATA_DIR = os.path.join(root_dir, os.getenv('DATA_DIR', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

print(f"🔧 데이터 디렉토리: {DATA_DIR}")

# ⏱️ 시간 간격 매핑
INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440
}

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
def update_data(symbol, interval, days, data_dir=None):
    # data_dir이 제공되지 않으면 현재 작업 디렉토리를 사용
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
        print(f"📂 기존 파일 발견: {filename}")
        print(f"📅 마지막 데이터: {last_date} → 이후 데이터만 수집")
        new_df = get_ohlcv(full_symbol, interval, start_time)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset='timestamp').sort_values('timestamp')
    else:
        print(f"📁 파일 없음 → 전체 기간 수집: {days}일")
        combined_df = get_ohlcv(full_symbol, interval, start_time)

    combined_df.to_csv(filename, index=False)
    print(f"✅ {full_symbol} 데이터 저장 완료: {filename}")
    print(f"🧮 총 데이터 수: {len(combined_df)}행")

    return combined_df  # ✅ 이 줄 추가

# 🧪 실행 예시
if __name__ == "__main__":
    # 예: BTC, 1시간 간격, 최근 365일
    coin_name = 'BTC'
    update_data(symbol=coin_name, interval='15m', days=365)
    update_data(symbol=coin_name, interval='1h', days=365)
    update_data(symbol=coin_name, interval='4h', days=365)
    update_data(symbol=coin_name, interval='1d', days=365)

    coin_name = 'ETH'
    update_data(symbol=coin_name, interval='15m', days=365)
    update_data(symbol=coin_name, interval='1h', days=365)
    update_data(symbol=coin_name, interval='4h', days=365)
    update_data(symbol=coin_name, interval='1d', days=365)

    coin_name = 'XRP'
    update_data(symbol=coin_name, interval='15m', days=365)
    update_data(symbol=coin_name, interval='1h', days=365)
    update_data(symbol=coin_name, interval='4h', days=365)
    update_data(symbol=coin_name, interval='1d', days=365)