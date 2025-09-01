## LSTM 기반 암호화폐 자동 트레이딩 봇

이 프로젝트는 바이낸스(Binance) 데이터를 활용하여 딥러닝 모델(LSTM)로 암호화폐 가격을 예측하고, 이를 기반으로 자동 트레이딩 전략을 실행하는 봇입니다. 🖥️ Tkinter GUI를 통해 봇의 실시간 상태, 거래 내역, 수익률 등을 시각적으로 확인할 수 있습니다.

-----

### 🚀 주요 기능

  * **다중 타임프레임 예측**: 15분, 1시간, 4시간, 1일 등 여러 타임프레임의 데이터를 기반으로 학습된 LSTM 모델을 활용하여 시장의 다양한 관점을 통합 예측합니다.
  * **유연한 외부 파일 관리**: `.exe` 실행 파일과 같은 위치에 `.env` 설정 파일, `/models` (학습된 모델), `/data` (시세 데이터) 폴더를 두고 사용할 수 있어 배포 및 관리가 용이합니다.
  * **실시간 GUI**: 현재 시세, 포지션, 누적 수익, 총 수익률, 매수/매도 횟수 등 봇의 핵심 정보를 한눈에 볼 수 있는 사용자 친화적인 그래픽 인터페이스를 제공합니다.
  * **가상 트레이딩 시뮬레이션**: 실제 자산 없이 가상의 자본으로 트레이딩 전략을 실행하고 성능을 평가할 수 있습니다.
  * **텔레그램 알림**: "강력 매수" 또는 "강력 매도" 신호가 발생할 경우, 설정된 텔레그램 채널로 실시간 알림을 전송하여 중요한 시장 변화를 놓치지 않도록 돕습니다. 🚨

-----

### 📁 폴더 및 파일 구조

프로젝트의 올바른 작동을 위해서는 다음과 같은 폴더 구조가 필요합니다. 특히 `trading_bot.exe` 파일이 실행되는 위치에 `.env` 파일과 `/models`, `/data`, `/utils` 폴더가 존재해야 합니다.

```
/ (프로젝트 루트 디렉토리)
├── trading_bot.exe        # 트레이딩 봇 실행 파일
├── .env                   # 환경 변수 설정 파일 (API 키 등)
├── /data                  # 코인별 OHLCV 시세 데이터 저장 폴더 (예: BTCUSDT_1d.csv)
│   └── (데이터 파일들)
├── /models                # 학습된 딥러닝 모델 파일 저장 폴더 (예: BTCUSDT_20250829_LSTM.pkl)
│   └── (모델 파일들)
└── /utils                 # 봇의 핵심 기능을 제공하는 모듈 폴더
    ├── __init__.py        # Python 패키지임을 알림
    ├── load_ohlcv.py      # OHLCV 데이터 로딩 및 관리
    ├── model_package.py   # 모델 패키징 (모델, 스케일러, 피처 정보)
    ├── telegram.py        # 텔레그램 메시지 전송 기능
    └── training.py        # 모델 학습 스크립트 (이 프로젝트는 이미 학습된 모델 사용)
```

-----

### ⚙️ 설치 및 실행 방법

#### 1\. 필수 라이브러리 설치

Python 환경에서 이 프로젝트를 실행하기 위해 필요한 라이브러리들을 설치해야 합니다. 다음 `requirements.txt` 파일을 프로젝트 루트 디렉토리에 생성하고 아래 내용을 추가한 후 `pip install -r requirements.txt` 명령어를 실행하세요.

````
# requirements.txt
python-binance
python-dotenv
pandas
numpy
scikit-learn
tensorflow # 또는 keras
joblib
requests
```sh
pip install -r requirements.txt
````

#### 2\. 설정 파일(`.env`) 준비

`trading_bot.exe` 파일이 위치한 **동일한 디렉토리**에 `.env` 파일을 생성하고 아래 예시와 같이 내용을 채워주세요.

  * `BINANCE_API_KEY`, `BINANCE_API_SECRET`: 바이낸스 API 키와 시크릿 키는 바이낸스 계정에서 발급받아야 합니다.
  * `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`: 텔레그램 봇을 생성하고 채널 ID를 얻어 입력합니다. 텔레그램 알림을 사용하지 않으려면 비워두거나 이 줄을 삭제해도 됩니다.

<!-- end list -->

```ini
# .env 파일 내용
BINANCE_API_KEY=YOUR_BINANCE_API_KEY_HERE
BINANCE_API_SECRET=YOUR_BINANCE_API_SECRET_HERE

TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_GROUP_ID_HERE
```

#### 3\. 모델 파일(`.pkl`) 및 데이터(`data`) 폴더 준비

  * `trading_bot.exe` 파일이 있는 **동일한 디렉토리**에 `models`라는 이름의 폴더를 생성하고, 학습된 `.pkl` 모델 파일들을 그 안에 넣어주세요. (예: `BTCUSDT_20250829_LSTM.pkl`).
  * 마찬가지로 `data`라는 이름의 폴더를 생성하고, 코인별 시세 데이터 CSV 파일들을 그 안에 넣어주세요. (예: `BTCUSDT_1d.csv`).

#### 4\. `.exe` 파일 실행

모든 준비가 완료되면, `trading_bot.exe` 파일을 더블 클릭하여 실행합니다. GUI 창이 나타나면 `봇 시작` 버튼을 눌러 트레이딩 봇을 작동시킬 수 있습니다.

-----

### 📝 PyInstaller로 `.exe` 파일 생성 (개발자용)

`trading_bot.py` 소스 코드를 `.exe` 실행 파일로 만들려면 `PyInstaller`를 사용합니다. 이때 `/utils` 폴더는 `.exe` 파일 내부에 포함시키고, `.env`, `/models`, `/data` 폴더는 외부에 두어 실행 시점에 해당 경로를 참조하도록 설정합니다.

프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요:

```sh
pyinstaller --onefile --add-data "utils;utils" --collect-data dateparser --hidden-import=sklearn.preprocessing._data trading_bot.py
```

  * `--onefile`: 단일 실행 파일로 패키징합니다.
  * `--add-data "utils;utils"`: `trading_bot.py`와 같은 위치에 있는 `utils` 폴더를 실행 파일 내부에 `utils`라는 이름으로 포함시킵니다.

이 명령어를 실행하면 `dist` 폴더 안에 `trading_bot.exe` 파일이 생성됩니다. 이 `.exe` 파일을 위에서 설명한 `models`, `data`, `.env` 파일과 함께 배치하여 사용하면 됩니다.