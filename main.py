import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 환경에 맞게 설정)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 또는 적합한 한글 폰트 경로로 설정
font_prop = fm.FontProperties(fname=font_path)

# MySQL 데이터베이스 연결
engine = create_engine('mysql+pymysql://root:kmo665100!!@localhost/crypto_db')

# 데이터 조회 쿼리
query_crypto = "SELECT * FROM crypto_data"
query_bitcoin_price = "SELECT * FROM bitcoin_price"

# 데이터 로드
crypto_df = pd.read_sql(query_crypto, engine)
bitcoin_price_df = pd.read_sql(query_bitcoin_price, engine)

# 'price'와 'volume' 컬럼이 문자열일 경우 숫자로 변환
crypto_df['price'] = pd.to_numeric(crypto_df['price'], errors='coerce')
crypto_df['volume'] = pd.to_numeric(crypto_df['volume'], errors='coerce')
bitcoin_price_df['price'] = pd.to_numeric(bitcoin_price_df['price'], errors='coerce')

# NaN 값 처리 (필요시)
crypto_df = crypto_df.dropna(subset=['price', 'volume'])  # NaN이 있는 행 제거
bitcoin_price_df = bitcoin_price_df.dropna(subset=['price'])

# 날짜 변환
crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
bitcoin_price_df['timestamp'] = pd.to_datetime(bitcoin_price_df['timestamp'])

# 첫 번째 그래프: 비트코인 가격
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('비트코인 가격', fontproperties=font_prop)  # 제목 한글 설정
bitcoin_price_df.set_index('timestamp')['price'].plot(label='비트코인 가격', ax=ax1, color='r')

ax1.set_xlabel('날짜', fontproperties=font_prop)  # x축 레이블 한글 설정
ax1.set_ylabel('가격', fontproperties=font_prop)  # y축 레이블 한글 설정
ax1.legend(loc='upper left', prop=font_prop)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 두 번째 그래프: 비트코인 시세 예측
# Linear Regression을 통한 시세 예측
bitcoin_price_df = bitcoin_price_df.dropna(subset=['price'])

# 날짜를 정수형 값으로 변환
bitcoin_price_df['timestamp_int'] = bitcoin_price_df['timestamp'].map(pd.Timestamp.toordinal)

# 훈련 데이터 준비
X = bitcoin_price_df[['timestamp_int']]
y = bitcoin_price_df['price']

# Linear Regression 모델 생성
model = LinearRegression()
model.fit(X, y)

# 향후 날짜 예측 (예: 2025년 1월 1일부터 7일까지)
future_dates = pd.date_range(start='2025-01-01', end='2025-01-07', freq='D')
future_dates_int = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

# 예측된 가격
predicted_prices = model.predict(future_dates_int)

# 예측 결과 그래프
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title('비트코인 시세 예측 (2025)', fontproperties=font_prop)  # 제목 한글 설정

# 예측된 가격을 플로팅
ax2.plot(future_dates, predicted_prices, label='예측된 가격', color='green')

ax2.set_xlabel('날짜', fontproperties=font_prop)  # x축 레이블 한글 설정
ax2.set_ylabel('예측 가격', fontproperties=font_prop)  # y축 레이블 한글 설정
ax2.legend(loc='upper left', prop=font_prop)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
