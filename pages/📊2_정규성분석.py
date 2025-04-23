import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def detect_outliers(data, method='IQR', threshold=1.5):
    """
    이상치를 탐지하는 함수
    
    Parameters:
    -----------
    data : array-like
        이상치를 탐지할 데이터
    method : str, optional
        이상치 탐지 방법 ('IQR' 또는 'Z-Score')
    threshold : float, optional
        이상치 판단 기준값
        
    Returns:
    --------
    outliers : array-like
        이상치 여부를 나타내는 불리언 배열
    """
    if method == 'IQR':
        # IQR 방법으로 이상치 탐지
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    else:  # Z-score 방법
        # Z-score 방법으로 이상치 탐지
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
    
    return outliers

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="정규성 분석",
    page_icon="📊",
    layout="wide"
)

# 그래프를 중앙에 표시하는 헬퍼 함수
def display_plot_centered(fig, width_pct=90):
    """Matplotlib 그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.pyplot(fig)

def display_plotly_centered(fig, width_pct=90):
    """Plotly 그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=True)

# Johnson 변환 함수
def johnson_transform(data):
    """Johnson 변환을 수행하는 함수"""
    try:
        from scipy.stats import johnsonsu
        # Johnson 변환 수행
        params = johnsonsu.fit(data)
        transformed_data = johnsonsu.cdf(data, *params)
        
        # 변환 정보 저장
        transform_info = {
            'gamma': params[0],  # shape 파라미터
            'delta': params[1],  # shape 파라미터
            'xi': params[2],     # location 파라미터
            'lambda': params[3],  # scale 파라미터
            'family': 'SU',      # Johnson 변환 family
            'formula': f"{params[0]} + {params[1]} * asinh((X - {params[2]}) / {params[3]})"
        }
        
        return transformed_data, transform_info
    except Exception as e:
        st.error(f"Johnson 변환 중 오류 발생: {str(e)}")
        return None, None

# Box-Cox 변환 함수
def box_cox_transform(data):
    """Box-Cox 변환을 수행하는 함수"""
    try:
        # 데이터가 모두 양수인지 확인
        if (data <= 0).any():
            # 음수나 0이 있는 경우, 모든 데이터를 양수로 만들기 위해 이동
            shift = abs(data.min()) + 1
            data_shifted = data + shift
        else:
            data_shifted = data
            
        # Box-Cox 변환 수행
        transformed_data, lambda_val = stats.boxcox(data_shifted)
        
        # 변환 정보 저장
        transform_info = {
            'lambda': lambda_val,
            'shift': shift if (data <= 0).any() else 0,
            'family': 'Box-Cox',
            'formula': f"(X^{lambda_val:.4f} - 1) / {lambda_val:.4f} if lambda != 0 else ln(X)"
        }
        
        return transformed_data, transform_info
    except Exception as e:
        st.error(f"Box-Cox 변환 중 오류 발생: {str(e)}")
        return None, None

# Log 변환 함수
def log_transform(data):
    """Log 변환을 수행하는 함수"""
    try:
        # 데이터가 모두 양수인지 확인
        if (data <= 0).any():
            # 음수나 0이 있는 경우, 모든 데이터를 양수로 만들기 위해 이동
            shift = abs(data.min()) + 1
            data_shifted = data + shift
        else:
            data_shifted = data
            
        # Log 변환 수행
        transformed_data = np.log(data_shifted)
        
        # 변환 정보 저장
        transform_info = {
            'shift': shift if (data <= 0).any() else 0,
            'family': 'Log',
            'formula': 'ln(X)'
        }
        
        return transformed_data, transform_info
    except Exception as e:
        st.error(f"Log 변환 중 오류 발생: {str(e)}")
        return None, None

# 변환 결과 시각화 함수
def show_transformation_comparison(original_data, transformed_data, var_name, transform_info=None):
    """원본 데이터와 변환된 데이터를 비교하여 시각화하는 함수"""
    
    # 원본 데이터 시각화
    st.subheader("1. 원본 데이터 분석")
    
    # 원본 데이터 서브플롯 생성
    fig_original = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "원본 데이터 확률도 (Q-Q Plot)",
            "원본 데이터 분포"
        ),
        horizontal_spacing=0.15
    )
    
    # 색상 정의
    COLORS = {
        'data_points': '#1f77b4',      # 데이터 포인트 (파란색)
        'reference_line': '#ff7f0e',    # 기준선 (주황색)
        'confidence': 'rgba(255, 127, 14, 0.2)',  # 신뢰구간 (연한 주황색)
        'normal_dist': '#2ca02c'        # 정규분포 (초록색)
    }
    
    # 원본 데이터 Q-Q Plot
    qq_data_orig = stats.probplot(original_data, dist="norm", fit=True)
    theoretical_quantiles_orig = qq_data_orig[0][0]
    sample_quantiles_orig = qq_data_orig[0][1]
    slope_orig, intercept_orig, r_orig = qq_data_orig[1]
    
    # 데이터 포인트
    fig_original.add_trace(
        go.Scatter(
            x=theoretical_quantiles_orig,
            y=sample_quantiles_orig,
            mode='markers',
            name='데이터',
            marker=dict(color=COLORS['data_points'], size=8),
            hovertemplate='이론적 분위수: %{x:.2f}<br>실제 분위수: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 기준선
    line_x = np.linspace(min(theoretical_quantiles_orig), max(theoretical_quantiles_orig), 100)
    line_y = slope_orig * line_x + intercept_orig
    fig_original.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name='기준선',
            line=dict(color=COLORS['reference_line'], width=2),
            hovertemplate='이론적 분위수: %{x:.2f}<br>예상 분위수: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 신뢰구간
    confidence_band = 1.96 * np.std(sample_quantiles_orig - (slope_orig * theoretical_quantiles_orig + intercept_orig))
    fig_original.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y + confidence_band,
            mode='lines',
            name='95% 신뢰구간',
            line=dict(color=COLORS['reference_line'], width=1, dash='dot'),
            showlegend=False
        ),
        row=1, col=1
    )
    fig_original.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y - confidence_band,
            mode='lines',
            name='95% 신뢰구간',
            line=dict(color=COLORS['reference_line'], width=1, dash='dot'),
            fill='tonexty',
            fillcolor=COLORS['confidence'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 원본 데이터 분포
    hist_orig, bins = np.histogram(original_data, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 정규분포 곡선
    x = np.linspace(min(original_data), max(original_data), 100)
    normal_pdf = stats.norm.pdf(x, np.mean(original_data), np.std(original_data))
    
    fig_original.add_trace(
        go.Histogram(
            x=original_data,
            name='히스토그램',
            histnorm='probability density',
            opacity=0.7,
            showlegend=True
        ),
        row=1, col=2
    )
    
    fig_original.add_trace(
        go.Scatter(
            x=x,
            y=normal_pdf,
            name='정규분포',
            line=dict(color=COLORS['normal_dist'], dash='dash'),
            mode='lines'
        ),
        row=1, col=2
    )
    
    # 레이아웃 업데이트
    fig_original.update_layout(
        height=400,
        showlegend=True,
        title_text=f"원본 데이터 분석 (R² = {r_orig**2:.4f})",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # 축 레이블 업데이트
    fig_original.update_xaxes(title_text="이론적 분위수", row=1, col=1)
    fig_original.update_yaxes(title_text="실제 분위수", row=1, col=1)
    fig_original.update_xaxes(title_text="데이터 값", row=1, col=2)
    fig_original.update_yaxes(title_text="밀도", row=1, col=2)
    
    # 그리드 추가
    fig_original.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig_original.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # 원본 데이터 그래프 표시
    display_plotly_centered(fig_original)
    
    # 변환된 데이터 시각화
    st.subheader(f"2. 변환된 데이터 분석 ({transform_info['family']})")
    
    # 변환된 데이터 서브플롯 생성
    fig_transformed = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "변환된 데이터 확률도 (Q-Q Plot)",
            "변환된 데이터 분포"
        ),
        horizontal_spacing=0.15
    )
    
    # 변환된 데이터 Q-Q Plot
    qq_data_trans = stats.probplot(transformed_data, dist="norm", fit=True)
    theoretical_quantiles_trans = qq_data_trans[0][0]
    sample_quantiles_trans = qq_data_trans[0][1]
    slope_trans, intercept_trans, r_trans = qq_data_trans[1]
    
    # 데이터 포인트
    fig_transformed.add_trace(
        go.Scatter(
            x=theoretical_quantiles_trans,
            y=sample_quantiles_trans,
            mode='markers',
            name='변환된 데이터',
            marker=dict(color=COLORS['data_points'], size=8),
            hovertemplate='이론적 분위수: %{x:.2f}<br>실제 분위수: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 기준선
    line_x = np.linspace(min(theoretical_quantiles_trans), max(theoretical_quantiles_trans), 100)
    line_y = slope_trans * line_x + intercept_trans
    fig_transformed.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name='기준선',
            line=dict(color=COLORS['reference_line'], width=2),
            hovertemplate='이론적 분위수: %{x:.2f}<br>예상 분위수: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 신뢰구간
    confidence_band = 1.96 * np.std(sample_quantiles_trans - (slope_trans * theoretical_quantiles_trans + intercept_trans))
    fig_transformed.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y + confidence_band,
            mode='lines',
            name='95% 신뢰구간',
            line=dict(color=COLORS['reference_line'], width=1, dash='dot'),
            showlegend=False
        ),
        row=1, col=1
    )
    fig_transformed.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y - confidence_band,
            mode='lines',
            name='95% 신뢰구간',
            line=dict(color=COLORS['reference_line'], width=1, dash='dot'),
            fill='tonexty',
            fillcolor=COLORS['confidence'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 변환된 데이터 분포
    hist_trans, bins = np.histogram(transformed_data, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 정규분포 곡선
    x = np.linspace(min(transformed_data), max(transformed_data), 100)
    normal_pdf = stats.norm.pdf(x, np.mean(transformed_data), np.std(transformed_data))
    
    fig_transformed.add_trace(
        go.Histogram(
            x=transformed_data,
            name='히스토그램',
            histnorm='probability density',
            opacity=0.7,
            showlegend=True
        ),
        row=1, col=2
    )
    
    fig_transformed.add_trace(
        go.Scatter(
            x=x,
            y=normal_pdf,
            name='정규분포',
            line=dict(color=COLORS['normal_dist'], dash='dash'),
            mode='lines'
        ),
        row=1, col=2
    )
    
    # 레이아웃 업데이트
    fig_transformed.update_layout(
        height=400,
        showlegend=True,
        title_text=f"변환된 데이터 분석 (R² = {r_trans**2:.4f})",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # 축 레이블 업데이트
    fig_transformed.update_xaxes(title_text="이론적 분위수", row=1, col=1)
    fig_transformed.update_yaxes(title_text="실제 분위수", row=1, col=1)
    fig_transformed.update_xaxes(title_text="데이터 값", row=1, col=2)
    fig_transformed.update_yaxes(title_text="밀도", row=1, col=2)
    
    # 그리드 추가
    fig_transformed.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig_transformed.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # 변환된 데이터 그래프 표시
    display_plotly_centered(fig_transformed)
    
    # 변환 정보 표시
    if transform_info:
        st.subheader("3. 변환 정보")
        
        # 변환 방법별 설명을 expander로 변경
        with st.expander("📚 변환 방법 설명", expanded=False):
            st.markdown("""
            **Johnson 변환 (SU)**
            - SU는 "Unbounded"의 약자로, 데이터의 범위에 제한이 없는 경우 사용
            - γ(gamma)와 δ(delta): 변환의 형태를 결정하는 모양 파라미터
            - ξ(xi): 데이터의 위치를 조정하는 위치 파라미터
            - λ(lambda): 데이터의 스케일을 조정하는 스케일 파라미터
            - 변환 공식: γ + δ * asinh((X - ξ) / λ)
            
            **Box-Cox 변환**
            - λ(lambda) 값에 따라 다양한 변환 수행
            - λ = 0: 로그 변환 (ln(X))
            - λ = 0.5: 제곱근 변환 (√X)
            - λ = 1: 변환 없음 (X)
            - λ = 2: 제곱 변환 (X²)
            
            **Log 변환**
            - 자연로그(ln)를 사용한 변환
            - 데이터의 범위가 클 때 효과적
            - 지수적으로 증가하는 데이터에 적합
            """)
        
        # 변환 파라미터와 변환 유형을 두 열로 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**변환 파라미터:**")
            if transform_info['family'] == 'SU':  # Johnson 변환
                st.write(f"""
                - γ (gamma): {transform_info['gamma']:.4f}
                  - 변환의 기본 형태를 결정
                  - 값이 클수록 변환 곡선이 더 가파름
                
                - δ (delta): {transform_info['delta']:.4f}
                  - 변환의 강도를 결정
                  - 양수: 오른쪽으로 치우친 데이터를 정규화
                  - 음수: 왼쪽으로 치우친 데이터를 정규화
                
                - ξ (xi): {transform_info['xi']:.4f}
                  - 데이터의 중심 위치를 조정
                  - 데이터의 평균값과 비슷한 값이 일반적
                
                - λ (lambda): {transform_info['lambda']:.4f}
                  - 변환의 스케일을 조정
                  - 값이 작을수록 변환이 더 강력함
                """)
            elif transform_info['family'] == 'Box-Cox':  # Box-Cox 변환
                st.write(f"""
                - λ (lambda): {transform_info['lambda']:.4f}
                  - 변환의 강도를 결정
                  - 0에 가까울수록 로그 변환에 가까움
                  - 1에 가까울수록 변환이 약함
                
                - 이동값: {transform_info['shift']:.4f}
                  - 음수나 0이 있는 경우 데이터를 양수로 만들기 위해 더한 값
                  - 변환 후에는 이 값을 빼서 원래 스케일로 복원 가능
                """)
            else:  # Log 변환
                st.write(f"""
                - 이동값: {transform_info['shift']:.4f}
                  - 음수나 0이 있는 경우 데이터를 양수로 만들기 위해 더한 값
                  - 변환 후에는 이 값을 빼서 원래 스케일로 복원 가능
                """)
            
            st.write("**변환 유형:**")
            st.write(f"""
            - {transform_info['family']}
              - SU: Johnson 변환 중 Unbounded 타입
              - Box-Cox: Box-Cox 변환
              - Log: 로그 변환
            
            - 변환 공식: {transform_info['formula']}
              - 실제 데이터에 적용되는 수학적 변환식
              - X는 원본 데이터를 의미
            """)
        
        with col2:
            st.write("**정규성 개선 효과:**")
            st.write(f"""
            - R² 값 비교:
              - 원본 데이터 R²: {r_orig**2:.4f}
                * 1에 가까울수록 정규성이 좋음
                * 0.95 이상이면 매우 좋은 정규성
              
              - 변환 데이터 R²: {r_trans**2:.4f}
                * 원본보다 높아졌다면 변환이 성공적
                * 0.95 이상이면 매우 좋은 정규성
            """)
            
            
            # 변환 효과 평가 (R² 값과 정규성 검정 결과 종합)
            r2_improvement = r_trans**2 - r_orig**2
            
            # 정규성 검정 수행
            transformed_stat, transformed_p = stats.shapiro(transformed_data)
            anderson_result = stats.anderson(transformed_data, dist='norm')
            
            # 정규성 만족 여부 판단 (p-value가 0.05보다 작으면 정규성을 만족하지 않음)
            is_normal_shapiro = transformed_p >= 0.05
            is_normal_anderson = anderson_result.statistic < anderson_result.critical_values[2]  # 5% 유의수준
            is_good_r2 = r_trans**2 >= 0.95

            st.write("\n**종합 평가:**")
            if r2_improvement > 0:
                if is_good_r2 and is_normal_shapiro:
                    st.success(f"""
                    ✅ 변환 결과: 매우 좋음
                    - R² 값이 {r2_improvement:.4f}만큼 개선되었습니다. (현재 R² = {r_trans**2:.4f})
                    - QQ-plot이 정규분포를 매우 잘 따르고 있습니다.
                    - 정규성 검정을 통과했습니다. (p-value = {transformed_p:.4f} ≥ 0.05)
                    
                    👉 결론: 변환된 데이터를 안심하고 사용하셔도 좋습니다.
                    """)
                elif is_good_r2:
                    st.warning(f"""
                    ⚠️ 변환 결과: 양호
                    - R² 값이 {r2_improvement:.4f}만큼 개선되었습니다. (현재 R² = {r_trans**2:.4f})
                    - QQ-plot은 정규분포를 잘 따르고 있으나,
                    - 정규성 검정을 통과하지 못했습니다. (p-value = {transformed_p:.4f} < 0.05)
                    
                    👉 결론: R² 값이 0.95 이상으로 양호하여, 실무적으로 사용 가능합니다.
                    단, 엄격한 정규성이 요구되는 분석에는 주의가 필요합니다.
                    """)
                elif is_normal_shapiro:
                    st.warning(f"""
                    ⚠️ 변환 결과: 보통
                    - R² 값이 {r2_improvement:.4f}만큼 개선되었습니다. (현재 R² = {r_trans**2:.4f})
                    - 정규성 검정을 통과했으나, (p-value = {transformed_p:.4f} ≥ 0.05)
                    - QQ-plot의 적합도가 다소 낮습니다.
                    
                    👉 결론: 정규성 검정은 통과했으나 R² 값이 낮아 주의가 필요합니다.
                    데이터의 특성과 분석 목적을 고려하여 사용 여부를 결정하세요.
                    """)
                else:
                    st.error(f"""
                    ❌ 변환 결과: 미흡
                    - R² 값이 {r2_improvement:.4f}만큼 개선되었으나, 여전히 낮습니다. (현재 R² = {r_trans**2:.4f})
                    - 정규성 검정을 통과하지 못했습니다. (p-value = {transformed_p:.4f} < 0.05)
                    
                    👉 결론: 다른 변환 방법을 시도해보시거나, 
                    비모수적 방법 사용을 고려하시기 바랍니다.
                    """)
            else:
                st.error(f"""
                ❌ 변환 효과 없음
                - R² 값이 {abs(r2_improvement):.4f}만큼 감소했습니다. (현재 R² = {r_trans**2:.4f})
                - 변환이 정규성을 개선하지 못했습니다.
                
                👉 결론: 
                1. 다른 변환 방법을 시도해보세요.
                2. 또는 원본 데이터를 그대로 사용하시기 바랍니다.
                3. 필요한 경우 비모수적 방법 사용을 고려하세요.
                """)
            
            # 변환 데이터 사용 여부 선택
            use_transformed = st.checkbox(
                "이 변환된 데이터를 이후 분석에 사용하시겠습니까?",
                help="체크하면 변환된 데이터가 공정능력분석 등에 사용됩니다."
            )
            
            if use_transformed:
                # 변환 정보 저장
                st.session_state.transformed_vars[selected_var] = {
                    'data': pd.Series(transformed_data, index=var_data.index),
                    'method': transform_info['family'],
                    'info': transform_info,
                    'r_squared': r_trans**2,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'original_data': var_data.copy()
                }
                
                # 변환된 데이터를 원본 데이터프레임에 저장
                if 'transformed_data' not in st.session_state:
                    st.session_state.transformed_data = st.session_state.data.copy()
                st.session_state.transformed_data[selected_var] = pd.Series(transformed_data, index=var_data.index)
                
                st.success(f"""
                ✅ 변환된 데이터가 저장되었습니다.
                - 변수: {selected_var}
                - 변환 방법: {transform_info['family']}
                - R² 값: {r_trans**2:.4f}
                """)
            else:
                # 변환 정보 제거
                if selected_var in st.session_state.transformed_vars:
                    del st.session_state.transformed_vars[selected_var]
                st.info("원본 데이터를 사용합니다.")
    
    # 변환된 데이터 다운로드 링크
    transformed_df = pd.DataFrame({
        '원본_데이터': original_data,
        '변환된_데이터': transformed_data
    })
    
    # UTF-8 BOM 인코딩으로 CSV 생성
    csv = transformed_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label=f"📥 {transform_info['family']} 변환 데이터 다운로드",
        data=csv,
        file_name=f"{var_name}_{transform_info['family']}_변환데이터.csv",
        mime="text/csv"
    )

# 변환된 데이터 관리를 위한 session state 초기화
if 'transformed_vars' not in st.session_state:
    st.session_state.transformed_vars = {}

# 메인 페이지
st.title("2. 정규성 분석")

# 정규성 분석 개념 설명
with st.expander("📚 정규성 분석이란?"):
    st.markdown("""
    ### 정규성 분석(Normality Analysis)
    정규성 분석은 데이터가 정규분포를 따르는지 확인하는 과정입니다. 많은 통계적 분석 방법들이 데이터의 정규성을 가정하므로, 
    정규성 검정은 매우 중요한 전처리 단계입니다.
    
    #### 주요 정규성 검정 방법
    1. **Shapiro-Wilk 검정**
       - 표본 크기가 작을 때(50개 미만) 주로 사용
       - 가장 널리 사용되는 정규성 검정 방법
       - p-value가 0.05보다 크면 정규분포를 따른다고 판단
    
    2. **Anderson-Darling 검정**
       - 표본 크기가 클 때도 사용 가능
       - 꼬리 부분의 차이에 더 민감
       - 정규성 검정의 정확도가 높은 방법
    
    #### 데이터 변환 방법
    데이터가 정규분포를 따르지 않을 경우, 다음과 같은 변환 방법을 사용할 수 있습니다:
    
    1. **Johnson 변환**
       - 다양한 형태의 비정규 분포를 정규분포로 변환
       - 가장 강력한 변환 방법 중 하나
       - SU(Unbounded), SB(Bounded), SL(Log-normal) 세 가지 변환 함수군 제공
       - 데이터의 형태에 따라 자동으로 최적의 변환 함수 선택
       - 해석: 변환 후 R² 값이 1에 가까울수록 정규성이 좋음
    
    2. **Box-Cox 변환**
       - λ(lambda) 값에 따라 다양한 변환 수행
       - λ = 0: 로그 변환
       - λ = 0.5: 제곱근 변환
       - λ = 1: 변환 없음
       - λ = 2: 제곱 변환
       - 해석: λ 값이 1에 가까울수록 원본 데이터가 정규성에 가까움
    
    3. **Log 변환**
       - 오른쪽으로 치우친(Positive Skewed) 분포를 정규분포에 가깝게 변환
       - 데이터의 범위가 클 때 효과적
       - 지수적으로 증가하는 데이터에 적합
       - 해석: 변환 후 분포가 대칭적이면 성공적인 변환
    
    #### 변환 결과 해석 방법
    1. **Q-Q Plot 해석**
       - 점들이 대각선(기준선)에 가까울수록 정규성이 좋음
       - 신뢰구간 안에 대부분의 점이 있어야 함
       - R² 값이 1에 가까울수록 정규성이 좋음
    
    2. **히스토그램 해석**
       - 종모양의 대칭적인 분포가 나타나면 정규성이 좋음
       - 빨간 정규분포 곡선과 히스토그램이 잘 일치해야 함
    
    3. **Shapiro-Wilk 검정 해석**
       - p-value > 0.05: 정규분포를 따름
       - p-value ≤ 0.05: 정규분포를 따르지 않음
    """)

# 데이터 확인
if 'data' in st.session_state and st.session_state.data is not None:
    data = st.session_state.data
    
    # 정규성 분석 설정
    st.subheader("정규성 분석 설정")
    
    # 변수 선택
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("분석할 변수 선택:", numeric_cols)
    
    if selected_var:
        # 선택된 변수의 데이터
        var_data = data[selected_var].dropna()
        
        if len(var_data) > 0:
            # 이상치 처리 옵션 추가
            st.subheader("이상치 처리")
            
            use_outlier_treatment = st.checkbox("이상치 처리 활성화", value=False, 
                                            help="변환 전에 데이터에서 이상치를 탐지하고 처리합니다.")

            if use_outlier_treatment:
                outlier_col1, outlier_col2 = st.columns(2)

                with outlier_col1:
                    outlier_method = st.selectbox(
                        "이상치 탐지 방법",
                        options=["IQR", "Z-Score"],
                        help="IQR: 사분위수 범위 기반 방법, Z-Score: 표준점수 기반 방법"
                    )
                    
                with outlier_col2:
                    if outlier_method == "IQR":
                        threshold = st.slider("IQR 임계값", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                        help="1.5(일반적 기준), 3.0(극단값만 탐지)")
                        st.caption("💡 임계값 1.5는 일반적인 기준, 3.0은 극단적인 이상치만 탐지")
                    else:  # Z-Score
                        threshold = st.slider("Z-Score 임계값", min_value=2.0, max_value=4.0, value=3.0, step=0.1,
                                        help="3.0(일반적 기준), 값이 클수록 극단적인 이상치만 탐지")
                        st.caption("💡 임계값 3.0은 데이터의 99.7%를 정상으로 간주 (정규분포 가정 시)")
                
                # 이상치 탐지
                outliers = detect_outliers(var_data, method=outlier_method, threshold=threshold)
                outlier_count = outliers.sum()
                
                # 이상치 정보 표시
                if outlier_count > 0:
                    st.info(f"탐지된 이상치: {outlier_count}개 ({outlier_count/len(var_data):.1%})")
                    
                    # 이상치 데이터 표시
                    if st.checkbox("이상치 데이터 보기"):
                        outlier_data = pd.DataFrame({
                            '값': var_data[outliers],
                            '원본 인덱스': var_data[outliers].index
                        }).reset_index(drop=True)
                        st.dataframe(outlier_data)
                        st.caption("⚠️ 위 이상치들은 분석에서 제외됩니다.")
                else:
                    st.success("이상치가 탐지되지 않았습니다.")
                
                # 이상치 제거
                if outlier_count > 0:
                    var_data = var_data[~outliers].copy()
                    st.warning(f"이상치 {outlier_count}개가 제거되었습니다. 남은 데이터: {len(var_data)}개")

            # 이전 변환 정보 표시
            if selected_var in st.session_state.transformed_vars:
                st.info(f"""
                ℹ️ 이 변수는 현재 {st.session_state.transformed_vars[selected_var]['method']} 변환이 적용되어 있습니다.
                - R² 값: {st.session_state.transformed_vars[selected_var]['r_squared']:.4f}
                - 변환 날짜: {st.session_state.transformed_vars[selected_var]['timestamp']}
                """)

            st.subheader("데이터 분포 시각화")
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "데이터 분포",
                    "Q-Q Plot"
                )
            )
            
            # 히스토그램과 정규분포 곡선
            hist, bins = np.histogram(var_data, bins='auto', density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # 정규분포 파라미터 계산
            mu, std = np.mean(var_data), np.std(var_data)
            x = np.linspace(min(var_data), max(var_data), 100)
            y = stats.norm.pdf(x, mu, std)
            
            # 히스토그램 추가
            fig.add_trace(
                go.Bar(x=bin_centers, y=hist, name="히스토그램", opacity=0.7),
                row=1, col=1
            )
            
            # 정규분포 곡선 추가
            fig.add_trace(
                go.Scatter(x=x, y=y, name="정규분포", line=dict(color='red')),
                row=1, col=1
            )
            
            # Q-Q Plot
            qq_data = stats.probplot(var_data, dist="norm", fit=True)
            theoretical_quantiles = qq_data[0][0]
            sample_quantiles = qq_data[0][1]
            slope, intercept, r = qq_data[1]
            
            # 데이터 포인트 추가
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='blue', size=8),
                    hovertemplate='이론적 분위수: %{x:.2f}<br>실제 분위수: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 참조선(직선) 추가
            line_x = np.linspace(min(theoretical_quantiles), max(theoretical_quantiles), 100)
            line_y = slope * line_x + intercept
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name='참조선',
                    line=dict(color='red', width=2, dash='solid'),
                    hovertemplate='이론적 분위수: %{x:.2f}<br>예상 분위수: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=400,
                showlegend=True,
                annotations=[
                    dict(
                        text=f"데이터 분포",
                        xref="paper", yref="paper",
                        x=0.25, y=1.0,
                        showarrow=False
                    ),
                    dict(
                        text=f"Q-Q Plot (R² = {r**2:.4f})",
                        xref="paper", yref="paper",
                        x=0.75, y=1.0,
                        showarrow=False
                    )
                ]
            )
            
            # 그래프 표시
            display_plotly_centered(fig)
            
            # 정규성 검정 수행
            shapiro_stat, shapiro_p = stats.shapiro(var_data)
            anderson_stat, anderson_crit, anderson_sig = stats.anderson(var_data, dist='norm')
            
            # Anderson-Darling 검정 결과를 더 명확하게 표시
            anderson_results = pd.DataFrame({
                '유의수준(%)': anderson_sig,
                '임계값': anderson_crit
            })
            
            # 정규성 검정 결과 표시
            st.subheader("정규성 검정 결과")
            
            # Shapiro-Wilk 검정 결과 표시
            st.markdown(f"""
            **1. Shapiro-Wilk 검정**
            - 통계량: {shapiro_stat:.4f}
            - p-value: {shapiro_p:.4f}
            - 결과: {'정규분포를 따름' if shapiro_p >= 0.05 else '정규분포를 따르지 않음'}
            """)
            
            # Anderson-Darling 검정 결과 표시
            st.markdown("""
            **2. Anderson-Darling 검정**
            - 통계량: {:.4f}
            - 결과: {}
            
            Anderson-Darling 검정의 임계값:
            """.format(
                anderson_stat,
                '정규분포를 따름' if anderson_stat < anderson_crit[2] else '정규분포를 따르지 않음'
            ))
            
            # Anderson-Darling 임계값 테이블 표시
            st.table(anderson_results.style.format("{:.4f}"))
            
            st.markdown("""
            **Anderson-Darling 검정 해석 방법:**
            - 통계량이 임계값보다 작으면 해당 유의수준에서 정규분포를 따릅니다.
            - 일반적으로 5% 유의수준(0.05)을 기준으로 판단합니다.
            - 현재 데이터는 5% 유의수준의 임계값 {:.4f}를 기준으로 판단했습니다.
            """.format(anderson_crit[2]))
            
            # 검정 결과에 대한 상세 설명
            st.markdown("""
            **검정 결과 해석:**
            
            1. **Shapiro-Wilk 검정**
               - p-value > 0.05: 정규분포를 따름
               - p-value ≤ 0.05: 정규분포를 따르지 않음
               - 표본 크기가 작을 때(50개 미만) 더 정확함
            
            2. **Anderson-Darling 검정**
               - 통계량 < 임계값: 정규분포를 따름
               - 통계량 ≥ 임계값: 정규분포를 따르지 않음
               - 큰 표본에서도 잘 작동하며, 꼬리 부분의 차이에 더 민감함
            """)
            
            # 두 검정 결과를 종합적으로 판단
            is_normal = (shapiro_p >= 0.05) and (anderson_stat < anderson_crit[2])
            
            if not is_normal:
                st.warning(f"""
                ⚠️ 정규성 검정 결과: 데이터가 정규분포를 따르지 않습니다.
                
                **해석 가이드:**
                - Shapiro-Wilk 검정: p-value = {shapiro_p:.4f}
                - Anderson-Darling 검정: 통계량 = {anderson_stat:.4f} (임계값 = {anderson_crit[2]:.4f})
                - Q-Q Plot에서 점들이 기준선에서 벗어나 있습니다.
                - 히스토그램이 정규분포 곡선과 차이가 있습니다.
                
                💡 데이터 변환을 통해 정규성을 개선할 수 있습니다.
                """)
            
            # 변환 옵션 제공
            transform_option = st.radio(
                "데이터 변환 방법 선택:",
                ["변환하지 않음", "Johnson 변환", "Box-Cox 변환", "Log 변환"]
            )
            
            if transform_option != "변환하지 않음":
                # 변환 수행
                if transform_option == "Johnson 변환":
                    transformed_data, transform_info = johnson_transform(var_data)
                elif transform_option == "Box-Cox 변환":
                    transformed_data, transform_info = box_cox_transform(var_data)
                else:  # Log 변환
                    transformed_data, transform_info = log_transform(var_data)
                
                if transformed_data is not None:
                    # 변환 결과 시각화 및 비교
                    show_transformation_comparison(var_data, transformed_data, selected_var, transform_info)
                else:
                    # 변환하지 않음 선택 시 변환 정보 제거
                    if selected_var in st.session_state.transformed_vars:
                        del st.session_state.transformed_vars[selected_var]
                    st.info("원본 데이터를 사용합니다.")
        else:
            st.error(f"선택한 변수 '{selected_var}'에 유효한 데이터가 없습니다.")
    else:
        st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")

# 페이지 하단 소개
st.markdown("---")
st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ") 