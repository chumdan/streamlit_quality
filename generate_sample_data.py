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

# 날짜 생성
dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
manufacturing_dates = random.sample(dates, n_samples)
manufacturing_dates.sort()

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

# 연속형 변수 생성
# 기본값 설정
base_yield = 95
base_hardness = 7.0
base_thickness = 7.5
base_compression = 11.0

# 계절성 효과 추가
def add_seasonal_effect(date):
    month = date.month
    if 6 <= month <= 8:  # 여름철
        return -2
    return 0

# 설비별 효과 추가
equipment_effects = {
    'Korsch#1': 1.0,
    'Korsch#2': 0.8,
    'Kilian#1': 1.2,
    'Kilian#2': 1.0,
    'Kilian#3': 0.9
}

# 작업조 효과 추가
shift_effects = {
    '주간조': 0.5,
    '야간조': -0.5
}

# 연속형 변수 생성
for i in range(n_samples):
    date = manufacturing_dates[i]
    equip = data['설비'][i]
    shift = data['작업조'][i]
    
    # 기본 효과 계산
    seasonal = add_seasonal_effect(date)
    equip_effect = equipment_effects[equip]
    shift_effect = shift_effects[shift]
    
    # 수율 생성 (기본값 + 효과 + 랜덤변동)
    yield_value = base_yield + seasonal + shift_effect + np.random.normal(0, 1)
    data.setdefault('수율(%)', []).append(max(80, min(105, yield_value)))
    
    # 경도 생성 (압축력과 양의 상관관계)
    hardness = base_hardness + (data['수율(%)'][-1] - base_yield) * 0.1 + np.random.normal(0, 0.3)
    data.setdefault('경도_Min(N)', []).append(max(5, min(9, hardness)))
    data.setdefault('경도_Max(N)', []).append(max(6, min(10, hardness + 1)))
    
    # 두께 생성
    thickness = base_thickness + np.random.normal(0, 0.2)
    data.setdefault('두께_Min', []).append(max(5, min(6, thickness)))
    data.setdefault('두께_Max', []).append(max(5.1, min(6.1, thickness + 0.1)))
    
    # 압축력 생성
    compression = base_compression * equip_effect + np.random.normal(0, 0.5)
    data.setdefault('압축력(KN)', []).append(max(8, min(15, compression)))
    
    # 입자 크기 생성 (서로 상관관계 있게)
    d01 = np.random.normal(0.6, 0.2)
    d05 = d01 * 8 + np.random.normal(0, 0.5)
    d09 = d05 * 2.5 + np.random.normal(0, 1)
    
    data.setdefault('d(0.1)', []).append(max(0.4, min(1.5, d01)))
    data.setdefault('d(0.5)', []).append(max(3, min(8, d05)))
    data.setdefault('d(0.9)', []).append(max(12, min(26, d09)))
    
    # 분해점, 분해시간 생성 (입자 크기와 양의 상관관계)
    disintegration = 0.1 + (d09 / 20) + np.random.normal(0, 0.05)
    disintegration_time = 15 + (d09 / 2) + np.random.normal(0, 2)
    
    data.setdefault('분해점(%)', []).append(max(0, min(0.3, disintegration)))
    data.setdefault('분해시간(min)', []).append(max(10, min(30, disintegration_time)))
    
    # 타블렛 관련 측정값 생성
    tablet_thickness = 7.5 + np.random.normal(0, 0.2)
    tablet_diameter = 2.5 + np.random.normal(0, 0.1)
    tablet_total = tablet_thickness + tablet_diameter * 2
    
    data.setdefault('타블렛 두께(mm)', []).append(max(6.5, min(8.5, tablet_thickness)))
    data.setdefault('타블렛 직경(mm)', []).append(max(2.2, min(2.8, tablet_diameter)))
    data.setdefault('타블렛 총두께(mm)', []).append(max(7, min(9, tablet_total)))
    
    # 타블렛 속도, 분해비 속도 생성
    tablet_speed = 90000 + np.random.normal(0, 5000)
    disintegration_speed = 30 + np.random.normal(0, 5)
    
    data.setdefault('타블렛 속도(rpm)', []).append(int(max(80000, min(100000, tablet_speed))))
    data.setdefault('분해비 속도(rpm)', []).append(int(max(20, min(60, disintegration_speed))))

# 이상치 추가 (약 5%의 데이터에 이상치 추가)
outlier_indices = random.sample(range(n_samples), int(n_samples * 0.05))
for idx in outlier_indices:
    # 수율 이상치 (극단적으로 낮은 값)
    if random.random() < 0.3:
        data['수율(%)'][idx] = random.uniform(60, 75)
    
    # 경도 이상치 (극단적으로 높은 값)
    if random.random() < 0.3:
        data['경도_Min(N)'][idx] = random.uniform(9.5, 12)
        data['경도_Max(N)'][idx] = random.uniform(10.5, 13)
    
    # 두께 이상치 (극단적으로 낮은 값)
    if random.random() < 0.3:
        data['두께_Min'][idx] = random.uniform(3, 4)
        data['두께_Max'][idx] = random.uniform(3.1, 4.1)
    
    # 압축력 이상치 (극단적으로 높은 값)
    if random.random() < 0.3:
        data['압축력(KN)'][idx] = random.uniform(16, 20)
    
    # 입자 크기 이상치 (극단적으로 높은 값)
    if random.random() < 0.3:
        data['d(0.1)'][idx] = random.uniform(1.6, 2.0)
        data['d(0.5)'][idx] = random.uniform(8.5, 10.0)
        data['d(0.9)'][idx] = random.uniform(27, 30)
    
    # 분해시간 이상치 (극단적으로 높은 값)
    if random.random() < 0.3:
        data['분해시간(min)'][idx] = random.uniform(31, 40)

# 정규성을 따르지 않는 변수 추가 (지수분포, 왜도가 큰 분포 등)
# 지수분포를 따르는 변수 (왜도가 큰 분포)
for i in range(n_samples):
    # 지수분포를 따르는 변수 (왜도가 큰 분포)
    exp_value = np.random.exponential(scale=2.0)
    data.setdefault('불순물_함량(mg)', []).append(round(exp_value, 2))
    
    # 이항분포를 따르는 변수 (이산형 분포)
    binom_value = np.random.binomial(n=10, p=0.3)
    data.setdefault('결함_개수', []).append(binom_value)
    
    # 균일분포를 따르는 변수 (정규분포가 아닌 분포)
    unif_value = np.random.uniform(low=0.5, high=2.5)
    data.setdefault('용해도(%)', []).append(round(unif_value, 2))
    
    # 혼합 분포 (정규분포와 지수분포의 혼합)
    if random.random() < 0.7:
        mix_value = np.random.normal(loc=5.0, scale=0.5)
    else:
        mix_value = np.random.exponential(scale=2.0) + 3.0
    data.setdefault('압축_강도(N)', []).append(round(mix_value, 2))

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
df['제조일자'] = df['제조일자'].dt.strftime('%Y.%m.%d')

# CSV 파일로 저장
df.to_csv('./프리그렐정_교육용.csv', index=False, encoding='utf-8-sig')

print("가상 데이터셋이 생성되었습니다.") 