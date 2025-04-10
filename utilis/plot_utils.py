import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_control_chart(data, mean, std):
    """관리도 생성 함수"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 데이터 포인트
    ax.plot(data, marker='o', linestyle='-', color='blue', markersize=4)
    
    # 관리한계선
    ax.axhline(y=mean, color='green', linestyle='-', label='평균')
    ax.axhline(y=mean + 3*std, color='red', linestyle='--', label='UCL (+3σ)')
    ax.axhline(y=mean - 3*std, color='red', linestyle='--', label='LCL (-3σ)')
    
    # 추가 관리선 (1σ, 2σ)
    ax.axhline(y=mean + 2*std, color='orange', linestyle=':', label='+2σ', alpha=0.5)
    ax.axhline(y=mean - 2*std, color='orange', linestyle=':', label='-2σ', alpha=0.5)
    ax.axhline(y=mean + std, color='orange', linestyle=':', label='+1σ', alpha=0.3)
    ax.axhline(y=mean - std, color='orange', linestyle=':', label='-1σ', alpha=0.3)
    
    # 그래프 설정
    ax.set_title('공정 관리도')
    ax.set_xlabel('관측치')
    ax.set_ylabel('측정값')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax

def create_process_capability_chart(data, mean, std, lsl, usl):
    """공정능력 분포도 생성 함수"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 히스토그램 및 커널밀도 그래프
    sns.histplot(data, kde=True, color='blue', ax=ax)
    
    # 관리선
    ax.axvline(x=mean, color='green', linestyle='-', label='평균')
    ax.axvline(x=mean + 3*std, color='red', linestyle='--', label='+3σ')
    ax.axvline(x=mean - 3*std, color='red', linestyle='--', label='-3σ')
    
    # 규격한계선
    ax.axvline(x=usl, color='purple', linestyle='-.', label='USL')
    ax.axvline(x=lsl, color='purple', linestyle='-.', label='LSL')
    
    # 그래프 설정
    ax.set_title('공정능력 분포도')
    ax.set_xlabel('측정값')
    ax.set_ylabel('빈도')
    ax.legend(loc='best')
    
    return fig, ax

def create_correlation_heatmap(data):
    """상관관계 히트맵 생성 함수"""
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('변수 간 상관관계 히트맵')
    plt.tight_layout()
    return plt.gcf()

def create_zscore_chart(z_scores):
    """Z-Score 시각화 함수"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 데이터 정렬
    z_df = pd.DataFrame({
        '변수': list(z_scores.keys()),
        'Z-Score': list(z_scores.values())
    })
    z_df = z_df.sort_values('Z-Score', key=abs, ascending=False)
    
    # 바 색상 설정 (|z| > 2인 경우 빨강, 아니면 파랑)
    colors = ['red' if abs(z) > 2 else 'blue' for z in z_df['Z-Score']]
    
    # 바 차트 그리기
    bars = ax.barh(z_df['변수'], z_df['Z-Score'], color=colors)
    
    # 기준선 추가
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.axvline(x=2, color='orange', linestyle='--', linewidth=0.8)
    ax.axvline(x=-2, color='orange', linestyle='--', linewidth=0.8)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=0.8)
    ax.axvline(x=-3, color='red', linestyle='--', linewidth=0.8)
    
    # 축 레이블 및 제목
    ax.set_xlabel('Z-Score')
    ax.set_title('변수별 Z-Score')
    ax.grid(True, linestyle='--', axis='x', alpha=0.7)
    
    return fig, ax