# utils/telegram.py
import os
import time
import requests
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 텔레그램 토큰과 그룹 ID를 환경 변수에서 로드
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GROUP_ID = os.getenv('TELEGRAM_GROUP_ID')

# 마지막 알림 시간을 저장하는 딕셔너리
last_alert_time = {}


def send_telegram_channel_message(symbol, message, level):
    """
    지정된 심볼에 대해 텔레그램 채널로 메시지를 전송한다.
    동일 심볼에 대해 일정 시간(5분) 내 중복 전송을 방지한다.
    """
    now_ts = time.time()
    cooldown = 300  # 5분

    # 심볼 정규화 (예: 대문자, 공백 제거)
    normalized_symbol = symbol.strip().upper()

    # 마지막 전송 후 쿨타임 이내면 알림 생략
    last_ts = last_alert_time.get(normalized_symbol)
    if last_ts and now_ts - last_ts < cooldown:
        print(f"⏳ 쿨타임 적용 중: {normalized_symbol} ({int(now_ts - last_ts)}초 경과)")
        return

    # 텔레그램 메시지 내용 구성
    full_msg = f"{level} {normalized_symbol}\n{message}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': GROUP_ID,
        'text': full_msg,
        'parse_mode': 'Markdown'
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        last_alert_time[normalized_symbol] = now_ts
        print(f"✅ 텔레그램 메시지 전송 완료: {normalized_symbol}")
    except Exception as e:
        print(f"❌ 텔레그램 메시지 전송 실패: {e}")