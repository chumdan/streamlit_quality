import pandas as pd
import numpy as np

def calculate_process_capability(data, lsl, usl):
    """공정능력지수 계산 함수"""
    mean = data.mean()
    std = data.std()
    
    # 공정능력지수 계산
    sigma = 6 * std
    cp = (usl - lsl) / sigma if sigma > 0 else 0
    cpu = (usl - mean) / (3 * std) if std > 0 else 0
    cpl = (mean - lsl) / (3 * std) if std > 0 else 0
    cpk = min(cpu, cpl)
    
    return {
        'mean': mean,
        'std': std,
        'cp': cp,
        'cpu': cpu,
        'cpl': cpl,
        'cpk': cpk
    }

def calculate_z_scores(data, batch_data):
    """Z-Score 계산 함수"""
    z_scores = {}
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if pd.notnull(batch_data[col]):
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            # 표준편차가 0이 아닌 경우에만 계산
            if col_std > 0:
                batch_value = batch_data[col]
                z_score = (batch_value - col_mean) / col_std
                z_scores[col] = z_score
    
    return z_scores