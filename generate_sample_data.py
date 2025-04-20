import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# 시드 설정
np.random.seed(42)
random.seed(42)

# 기본 설정
n_samples = 200
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

# 범주형 변수 정의
equipment = ['Korsch#1', 'Korsch#2', 'Kilian#1', 'Kilian#2', 'Kilian#3']
products = ['프리그렐정', '아스피린정', '세프트리악손정', '메트포르민정']
lines = ['1라인', '2라인', '3라인']
shifts = ['주간조', '야간조']

# 제조라인별 수율 기본 설정
line_yield_params = {
    '1라인': {'mean': 98.0, 'std': 1.5},   # 표준편차 증가
    '2라인': {'mean': 95.0, 'std': 1.5},   # 표준편차 증가
    '3라인': {'mean': 92.0, 'std': 1.5}    # 표준편차 증가
}

# 날짜 생성
dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
manufacturing_dates = dates.tolist()

# 제조번호 생성
def generate_lot_number(date, index):
    prefix = 'JE' if 'Kilian' in equipment[index % len(equipment)] else 'GE'
    return f"{prefix}{date.strftime('%y%m%d')}{index:03d}"

# 데이터 생성
data = {
    '제조번호': [generate_lot_number(date, i) for i, date in enumerate(manufacturing_dates)],
    '제조일자': manufacturing_dates,
    '설비': [random.choice(equipment) for _ in range(n_samples)],
    '제품명': [random.choice(products) for _ in range(n_samples)],
    '제조라인': [random.choice(lines) for _ in range(n_samples)],
    '작업조': [random.choice(shifts) for _ in range(n_samples)]
}

# 기본 설정값 수정
base_values = {
    'yield': 95,
    'hardness': 7.0,
    'thickness': 7.5,
    'compression': 11.0,
    'tablet_speed': 90000,
    'disintegration': 30
}

# 변수별 상관관계 가중치 (각 변수마다 다른 상관관계 수준 설정)
var_weights = {
    '압축력(KN)': 0.93,          # 매우 강한 양의 상관관계
    '경도_Min(N)': 0.85,         # 강한 양의 상관관계
    '경도_Max(N)': 0.82,         # 강한 양의 상관관계
    'd(0.1)': 0.65,              # 중간 수준의 양의 상관관계
    'd(0.5)': 0.76,              # 강한 양의 상관관계
    'd(0.9)': 0.7,               # 중간-강한 양의 상관관계
    '타블렛 속도(rpm)': 0.87,     # 강한 양의 상관관계
    '분해비 속도(rpm)': 0.79,     # 강한 양의 상관관계
    '타블렛 두께(mm)': -0.88,     # 강한 음의 상관관계
    '분해시간(min)': -0.73        # 중간-강한 음의 상관관계
}

# 수율 데이터 생성
yields = []
for i in range(n_samples):
    line = data['제조라인'][i]
    params = line_yield_params[line]
    
    # 기본 수율 생성
    base_yield = np.random.normal(params['mean'], params['std'])
    
    # 설비 및 작업조 효과 추가
    equip = data['설비'][i]
    shift = data['작업조'][i]
    
    if equip == 'Korsch#1':
        base_yield += 1.5
    elif equip == 'Korsch#2':
        base_yield -= 0.5
    elif equip == 'Kilian#1':
        base_yield += 1.0
    elif equip == 'Kilian#3':
        base_yield -= 1.0
        
    if shift == '주간조':
        base_yield += 0.8
    else:
        base_yield -= 0.8
        
    yields.append(base_yield)

# 이상치 추가 (15%의 데이터)
outlier_indices = random.sample(range(n_samples), int(n_samples * 0.15))
for idx in outlier_indices:
    if random.random() < 0.3:  # 극단적 이상치
        yields[idx] = yields[idx] + random.choice([-5, 5]) * random.uniform(0.8, 1.2)
    elif random.random() < 0.6:  # 중간 이상치
        yields[idx] = yields[idx] + random.choice([-3, 3]) * random.uniform(0.8, 1.2)
    else:  # 약한 이상치
        yields[idx] = yields[idx] + random.choice([-1.5, 1.5]) * random.uniform(0.8, 1.2)

data['수율(%)'] = yields

# 다른 변수들 생성
for i in range(n_samples):
    yield_value = yields[i]
    base_noise = np.random.normal(0, 0.5)  # 기본 노이즈 증가
    
    # 압축력 생성
    noise = np.random.normal(0, 0.8)  # 변수별 노이즈
    compression = 11.0 + (yield_value - 95) * 0.3 + noise
    data.setdefault('압축력(KN)', []).append(max(8, min(15, compression)))
    
    # 경도 생성
    noise = np.random.normal(0, 0.5)
    hardness_min = 7.0 + (yield_value - 95) * 0.2 + noise
    data.setdefault('경도_Min(N)', []).append(max(5, min(9, hardness_min)))
    
    noise = np.random.normal(0, 0.3)
    hardness_max = hardness_min + 0.5 + noise
    data.setdefault('경도_Max(N)', []).append(max(6, min(10, hardness_max)))
    
    # 입자 크기 생성
    noise = np.random.normal(0, 0.2)
    d01 = 0.6 + (yield_value - 95) * 0.02 + noise
    data.setdefault('d(0.1)', []).append(max(0.4, min(1.5, d01)))
    
    noise = np.random.normal(0, 0.5)
    d05 = 5.0 + (yield_value - 95) * 0.1 + noise
    data.setdefault('d(0.5)', []).append(max(3, min(8, d05)))
    
    noise = np.random.normal(0, 1.0)
    d09 = 15.0 + (yield_value - 95) * 0.2 + noise
    data.setdefault('d(0.9)', []).append(max(12, min(26, d09)))
    
    # 타블렛 속도 생성
    noise = np.random.normal(0, 2000)
    speed = 90000 + (yield_value - 95) * 1000 + noise
    data.setdefault('타블렛 속도(rpm)', []).append(int(max(80000, min(100000, speed))))
    
    # 분해비 속도 생성
    noise = np.random.normal(0, 5)
    disint_speed = 30 + (yield_value - 95) * 2 + noise
    data.setdefault('분해비 속도(rpm)', []).append(int(max(20, min(60, disint_speed))))
    
    # 타블렛 두께 생성
    noise = np.random.normal(0, 0.2)
    thickness = 7.5 - (yield_value - 95) * 0.05 + noise
    data.setdefault('타블렛 두께(mm)', []).append(max(6.5, min(8.5, thickness)))
    
    # 분해 시간 생성
    noise = np.random.normal(0, 2)
    disint_time = 20 - (yield_value - 95) * 0.5 + noise
    data.setdefault('분해시간(min)', []).append(max(10, min(30, disint_time)))

# 추가 이상치 생성 (다른 변수들에 대해)
for idx in random.sample(range(n_samples), int(n_samples * 0.1)):  # 10%의 데이터에 이상치 추가
    # 랜덤하게 1-3개의 변수 선택
    vars_to_modify = random.sample(list(var_weights.keys()), random.randint(1, 3))
    
    for var in vars_to_modify:
        current_value = data[var][idx]
        # 20% 확률로 극단적 이상치, 30% 확률로 중간 이상치, 50% 확률로 약한 이상치
        r = random.random()
        if r < 0.2:
            factor = random.uniform(1.3, 1.5)  # 극단적 이상치
        elif r < 0.5:
            factor = random.uniform(1.15, 1.3)  # 중간 이상치
        else:
            factor = random.uniform(1.05, 1.15)  # 약한 이상치
            
        if random.random() < 0.5:  # 50% 확률로 증가 또는 감소
            data[var][idx] = current_value * factor
        else:
            data[var][idx] = current_value / factor

# 수율 등급 생성
def get_yield_grade(yield_value):
    if yield_value >= 95:
        return 'A등급'
    elif yield_value >= 90:
        return 'B등급'
    else:
        return 'C등급'

data['수율_등급'] = [get_yield_grade(y) for y in data['수율(%)']]

# DataFrame 생성
df = pd.DataFrame(data)

# 날짜 형식 변환
df['제조일자'] = pd.to_datetime(df['제조일자']).dt.strftime('%Y.%m.%d')

# CSV 파일로 저장
df.to_csv('./프리그렐정_교육용.csv', index=False, encoding='utf-8-sig')

print("교육용 데이터가 생성되었습니다. 수율(%) 및 다른 변수들에 이상치가 추가되었습니다.")