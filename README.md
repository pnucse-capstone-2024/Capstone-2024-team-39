## 1. 프로젝트 소개

### 1-1. 배경 및 필요성
![image](https://github.com/user-attachments/assets/b1dbb31b-cbdc-4a97-b3d6-d251bd20ba28)

전력 피크 시간대에 전력 수요가 급격히 증가하고, 비피크 시간대에는 전력 소비 감축으로 전력계통이 불안정한 모습을 보이고 있다. 
이러한 문제를 해결하기 위해 **V2G(Vehicle to Grid)** 기술이 주목을 받고 있다. 
V2G는 자동차에서 전력망으로 전기를 이동하는 것을 의미하는데, 전기차에 저장된 배터리를 ‘에너지 저장 장치’로 활용하여 전력계통에 연계하는 기술을 의미한다.
전기차(EV) 시장의 급속한 성장으로 안전하고 신뢰성 높은 충전 인프라의 필요성이 대두되며, 피크 시간대의 전력 수요 증가와 사용자의 충전 요구 사항 (SoC, State of Charge)이 중요해지고 있다.
다만, 비피크 시간대에 모든 차량이 거의 동시에 충전되면 전력 수요가 급증한다는 단점이 있다.
따라서 전기차의 충방전을 최적화하기 위해 효율적이고 경제적인 방식으로 충전 전력을 관리하는 스마트 충전소의 출현이 요구된다. 스마트 충전소는 전기차가 충전 및 방전하는 시간대를 조절함으로써 피크 수요를 완화하고, 전력망의 안정성을 높여 전력 보조 서비스로서의 기능을 할 것으로 기대된다.

### 1-2. 목표 및 주요 내용
#### 1. V2G 이해관계자 별 이익을 고려한 상태 및 보상 함수 설계를 통한 이익 극대화
#### 2. 충방전 알고리즘 통합 상태 및 보상 함수 알고리즘을 통한 V2G 이해관계 최적화
#### 3. 전기차 충전 요금을 고려한 충방전 알고리즘 기반의 사용자 이익 극대화

---

## 2. 상세 설계

### 2-1. 시스템 구성도
![image](https://github.com/user-attachments/assets/f2029f4f-ac8e-4df0-9d7b-c78f5573af53)
1. V2G 이해관계자 별 상태 및 보상 함수 설계
2. EV2Gym 환경에서 시뮬레이션
3. 시뮬레이션 결과 pkl 파일 csv로 변환
4. Flask에서 충전소 별 충방전 상태 및 사용자 최대 이익 결과 시각화

### 2-2. 사용 기술
- Python - python 3.12.4
- Numpy - numpy 1.26.4
- Flask - flask 3.0.3
- Wandb - wandb 0.17.5
  
---

## 3. 설치 및 사용 방법

### 3-1. EV2Gym 실행 전, 기본 환경 설정
```
pip install ev2gym
```
```
pip install stable_baselines3
pip install sb3_contrib
pip install wandb
```

### 3-2. EV2Gym 환경으로 RL 학습 및 평가하기
1. 관리자 권한으로 prompt 실행
2. RL 학습하기
```
python run_RL_exp.py --config_file ./example_config_files/V2GProfitMax.yaml
```
(run_RL_exp.py 내에서 학습할 RL 알고리즘 설정 가능)
3. 시뮬레이션 평가하기
```
python run_evaluator_exp.py --config_file ./example_config_files/V2GProfitMax.yaml
```
(evaluator.py 내에서 평가할 RL 알고리즘 설정 가능)

### 3-3. 로컬 웹 구동하기
```
pip install flask pandas
```
1. 가상환경 생성하기
```
conda activate env
```
2. ev2gym/flask_visuals 내에서 flask app 실행하기
```
python app.py
```
3. 생성된 로컬 서버 내에서 이용 일자 및 충전소 선택하기
(이용 일자 : 23/10/03 ~ 23/10/09 선택 가능)
![image](https://github.com/user-attachments/assets/0a4e9969-9783-4a3f-bc9a-9e354a78e4f5)

4. 시뮬레이션 결과 확인하기 
![image](https://github.com/user-attachments/assets/3e945f9f-06b6-4728-857d-108588b466e9)

---

## 4. 소개 및 시연 영상
[![2024년 전기 졸업과제 39 EnerV2Gize] (https://img.youtube.com/vi/KgGFroZ9M_4/0.jpg)](https://www.youtube.com/watch?v=KgGFroZ9M_4&list=PLFUP9jG-TDp-CVdTbHvql-WoADl4gNkKj&index=38)


## 5. 팀 소개
- 이선진 (sunjin1101@pusan.ac.kr)
    - 알고리즘 설계, 모델 평가, flask 웹 시각화

- 이지은 (ljieun3004@pusan.ac.kr)
    - 알고리즘 설계, 결과 분석, flask 웹 시각화
