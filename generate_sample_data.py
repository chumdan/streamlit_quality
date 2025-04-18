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

# 제조라인별 수율 기본 설정 (더욱 안정적인 데이터)
line_yield_params = {
    '1라인': {'mean': 98.0, 'std': 0.5},   # 표준편차 더 감소
    '2라인': {'mean': 95.0, 'std': 0.5},   # 표준편차 더 감소
    '3라인': {'mean': 92.0, 'std': 0.5}    # 표준편차 더 감소
}

# 날짜 생성 (시계열 패턴을 위해 균등 분포)
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

# 계절성 효과 제거 (더 이상 사용하지 않음)
def add_seasonal_effect(date):
    return 0

# 시간에 따른 추세 효과 제거 (더 이상 사용하지 않음)
def add_trend_effect(date, start_date):
    return 0

# 설비별 효과 추가 (더욱 뚜렷한 차이)
equipment_effects = {
    'Korsch#1': {'yield': 1.5, 'hardness': 1.3, 'compression': 1.4},   # 효과 크게 증가
    'Korsch#2': {'yield': 0.7, 'hardness': 0.8, 'compression': 0.7},   # 효과 크게 감소
    'Kilian#1': {'yield': 1.3, 'hardness': 1.2, 'compression': 1.3},   # 효과 증가
    'Kilian#2': {'yield': 1.0, 'hardness': 1.0, 'compression': 1.0},   # 기준
    'Kilian#3': {'yield': 0.6, 'hardness': 0.7, 'compression': 0.6}    # 효과 크게 감소
}

# 작업조 효과 추가 (더욱 뚜렷한 차이)
shift_effects = {
    '주간조': {'yield': 1.5, 'hardness': 0.5, 'compression': 0.6},     # 효과 크게 증가
    '야간조': {'yield': -1.5, 'hardness': -0.5, 'compression': -0.6}   # 효과 크게 감소
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

# 변수 간 상관관계를 위한 기본 노이즈 생성 (더욱 최소화)
base_noise = np.random.normal(0, 0.01, n_samples)  # 노이즈 표준편차를 0.01로 더 감소

# 수율을 직접 생성
yields = {}
for line in lines:
    params = line_yield_params[line]
    # 정규분포로 생성
    yields[line] = np.random.normal(params['mean'], params['std'], n_samples)
    # 극단값 제한
    yields[line] = np.clip(yields[line], 
                          params['mean'] - 2*params['std'],
                          params['mean'] + 2*params['std'])

# 장비별 수율 효과 미리 계산
equipment_yield_effect = {
    'Korsch#1': 1.5,  # 매우 긍정적
    'Korsch#2': 0.5,  # 약간 부정적
    'Kilian#1': 1.2,  # 긍정적
    'Kilian#2': 1.0,  # 중립
    'Kilian#3': 0.3   # 매우 부정적
}

# 작업조별 수율 효과 미리 계산
shift_yield_effect = {
    '주간조': 0.8,    # 긍정적
    '야간조': -0.8    # 부정적
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

for i in range(n_samples):
    date = manufacturing_dates[i]
    equip = data['설비'][i]
    shift = data['작업조'][i]
    line = data['제조라인'][i]
    
    # 수율 계산 (기본값에 설비 및 작업조 효과 추가)
    base_yield_value = yields[line][i]
    yield_value = base_yield_value + equipment_yield_effect[equip] + shift_yield_effect[shift]
    
    # 극단값 제한
    yield_value = np.clip(yield_value,
                         line_yield_params[line]['mean'] - 2*line_yield_params[line]['std'],
                         line_yield_params[line]['mean'] + 2*line_yield_params[line]['std'])
    
    # 수율 저장
    data.setdefault('수율(%)', []).append(yield_value)
    
    # 정규화된 수율 (각 변수의 상관관계 계산에 사용)
    yield_normalized = (yield_value - line_yield_params[line]['mean']) / line_yield_params[line]['std']
    
    # 압축력 생성 (매우 강한 양의 상관관계: 목표 0.93)
    noise_factor = 0.005  # 매우 낮은 노이즈
    weight = var_weights['압축력(KN)']
    compression_value = base_values['compression'] + yield_normalized * weight * 3.0 + base_noise[i] * noise_factor
    data.setdefault('압축력(KN)', []).append(max(8, min(15, compression_value)))
    
    # 경도 생성 (강한 양의 상관관계: 목표 0.85)
    weight = var_weights['경도_Min(N)']
    hardness_value = base_values['hardness'] + yield_normalized * weight * 2.5 + base_noise[i] * noise_factor
    data.setdefault('경도_Min(N)', []).append(max(5, min(9, hardness_value)))
    
    weight = var_weights['경도_Max(N)']
    hardness_max_value = hardness_value + 0.5 + yield_normalized * weight * 0.1 + base_noise[i] * noise_factor
    data.setdefault('경도_Max(N)', []).append(max(6, min(10, hardness_max_value)))
    
    # 입자 크기 생성 (다양한 수준의 양의 상관관계)
    weight = var_weights['d(0.1)']
    d01_base = 0.6 + yield_normalized * weight * 0.4 + base_noise[i] * noise_factor
    
    weight = var_weights['d(0.5)']
    d05_base = 5.0 + yield_normalized * weight * 1.5 + base_noise[i] * noise_factor
    
    weight = var_weights['d(0.9)']
    d09_base = 15.0 + yield_normalized * weight * 2.5 + base_noise[i] * noise_factor
    
    data.setdefault('d(0.1)', []).append(max(0.4, min(1.5, d01_base)))
    data.setdefault('d(0.5)', []).append(max(3, min(8, d05_base)))
    data.setdefault('d(0.9)', []).append(max(12, min(26, d09_base)))
    
    # 타블렛 속도 생성 (강한 양의 상관관계: 목표 0.87)
    weight = var_weights['타블렛 속도(rpm)']
    tablet_speed = base_values['tablet_speed'] + yield_normalized * weight * 10000 + base_noise[i] * 50
    data.setdefault('타블렛 속도(rpm)', []).append(int(max(80000, min(100000, tablet_speed))))
    
    # 분해비 속도 생성 (강한 양의 상관관계: 목표 0.79)
    weight = var_weights['분해비 속도(rpm)']
    disintegration_speed = base_values['disintegration'] + yield_normalized * weight * 15 + base_noise[i] * 0.1
    data.setdefault('분해비 속도(rpm)', []).append(int(max(20, min(60, disintegration_speed))))
    
    # 타블렛 두께 생성 (강한 음의 상관관계: 목표 -0.88)
    weight = var_weights['타블렛 두께(mm)']
    tablet_thickness = base_values['thickness'] - yield_normalized * abs(weight) * 0.7 + base_noise[i] * noise_factor
    data.setdefault('타블렛 두께(mm)', []).append(max(6.5, min(8.5, tablet_thickness)))
    
    # 분해 시간 생성 (중간-강한 음의 상관관계: 목표 -0.73)
    weight = var_weights['분해시간(min)']
    disintegration_time = 20 - yield_normalized * abs(weight) * 8 + base_noise[i] * 0.1
    data.setdefault('분해시간(min)', []).append(max(10, min(30, disintegration_time)))

# 이상치는 최소화 (1% 미만으로 설정)
outlier_indices = random.sample(range(n_samples), int(n_samples * 0.01))
for idx in outlier_indices:
    if random.random() < 0.2:  # 이상치 생성 확률 20%로 감소
        current_line = data['제조라인'][idx]
        current_mean = line_yield_params[current_line]['mean']
        
        # 매우 작은 이상치만 추가
        if random.random() < 0.5:
            data['수율(%)'][idx] = current_mean + random.uniform(0.2, 0.5)
        else:
            data['수율(%)'][idx] = current_mean - random.uniform(0.2, 0.5)

# 정규성을 따르지 않는 변수 추가
for i in range(n_samples):
    # 불순물 함량 (지수분포, 압축력과 약한 음의 상관관계)
    exp_base = 2.0 - data['압축력(KN)'][i] * 0.1 + base_noise[i] * 0.2
    exp_value = np.random.exponential(scale=max(0.5, exp_base))
    data.setdefault('불순물_함량(mg)', []).append(round(exp_value, 2))
    
    # 결함 개수 (이항분포, 수율과 음의 상관관계)
    p_defect = 0.3 + (100 - data['수율(%)'][i]) * 0.01
    binom_value = np.random.binomial(n=10, p=min(0.9, max(0.1, p_defect)))
    data.setdefault('결함_개수', []).append(binom_value)
    
    # 용해도 (균일분포, 입자 크기와 음의 상관관계)
    unif_base = 2.5 - data['d(0.9)'][i] * 0.05
    unif_value = np.random.uniform(low=0.5, high=max(0.6, unif_base))
    data.setdefault('용해도(%)', []).append(round(unif_value, 2))
    
    # 압축 강도 (혼합 분포, 압축력과 강한 양의 상관관계)
    strength_base = data['압축력(KN)'][i] * 0.5
    if random.random() < 0.7:
        mix_value = np.random.normal(loc=strength_base, scale=0.5)
    else:
        mix_value = np.random.exponential(scale=strength_base/3) + strength_base
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

print("예측 모델링이 가능한 가상 데이터셋이 생성되었습니다.") 