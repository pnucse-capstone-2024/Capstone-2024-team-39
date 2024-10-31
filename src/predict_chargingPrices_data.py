import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import os

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv('electricity_prices.csv')
df['Datetime'] = pd.to_datetime(df['Datetime (UTC)'])
df.set_index('Datetime', inplace=True)
df['Price (EUR/MWhe)'] = df['Price (EUR/MWhe)'].clip(lower=0)  # 음수 값을 0으로 대체
df = df[['Price (EUR/MWhe)']]


#SARIMA 모델 설정 및 학습
train_data = df[:'2023']  # 2023년까지의 데이터를 학습에 사용
sarima_model = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 0, 24))
sarima_fit = sarima_model.fit(disp=False)

# 모델 요약 정보 출력
print(sarima_fit.summary())

#사용자 지정 시작 날짜 및 종료 날짜 설정
start_date_input = input("예측을 시작할 날짜를 입력하세요 (예: 2024-02-01): ")
end_date_input = input("예측을 종료할 날짜를 입력하세요 (예: 2024-02-05): ")

try:
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
except ValueError:
    print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력하세요.")
    exit()

# 날짜 범위에 대해 예측 수행
date_range = pd.date_range(start=start_date, end=end_date)

# 가격 데이터 저장 경로
save_path = 'pricedata'
os.makedirs(save_path, exist_ok=True)  # pricedata 폴더가 없으면 생성

for user_date in date_range:
    forecast_end = user_date + pd.Timedelta(days=1)
    total_hours = (forecast_end - df.index[-1]).days * 24

    if total_hours <= 0:
        print(f"{user_date.date()}는 데이터의 마지막 날짜 이후입니다.")
        continue

    forecast = sarima_fit.forecast(steps=total_hours)
    forecasted_date = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=total_hours, freq='H')
    forecast_series = pd.Series(forecast, index=forecasted_date)

    # 예측 데이터 출력
    start_of_day = user_date
    end_of_day = user_date + pd.Timedelta(hours=23)
    predicted_prices = forecast_series[start_of_day:end_of_day]

    if predicted_prices.empty:
        print(f"해당 날짜({user_date.date()})에 대한 예측 데이터가 없습니다.")
    else:
        print(f"{user_date.date()}의 시간대별 예상 전기 요금 (EUR/MWhe):\n{predicted_prices}")
        
        # 'Price (EUR/MWhe)' 컬럼 이름으로 데이터프레임 변환
        predicted_prices_df = predicted_prices.to_frame(name='Price (EUR/MWhe)')

        # 'Datetime (UTC)' 컬럼을 추가하고 인덱스 리셋
        predicted_prices_df.index.name = 'Datetime (UTC)'
        predicted_prices_df.reset_index(inplace=True)

        # 데이터 저장
        predicted_prices_df.to_csv(os.path.join(save_path, f'processed_electricity_prices_{user_date.date()}.csv'), index=False)
        
        # 예측 결과 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_prices.index, predicted_prices, label='Predicted Prices', color='green')
        plt.title(f'Electricity Prices on {user_date.date()}')
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/MWhe)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
