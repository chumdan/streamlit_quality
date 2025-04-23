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
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**변환 파라미터:**")
            if transform_info['family'] == 'SU':  # Johnson 변환
                st.write(f"- γ (gamma): {transform_info['gamma']:.4f}")
                st.write(f"- δ (delta): {transform_info['delta']:.4f}")
                st.write(f"- ξ (xi): {transform_info['xi']:.4f}")
                st.write(f"- λ (lambda): {transform_info['lambda']:.4f}")
            elif transform_info['family'] == 'Box-Cox':  # Box-Cox 변환
                st.write(f"- λ (lambda): {transform_info['lambda']:.4f}")
                if transform_info['shift'] > 0:
                    st.write(f"- 이동값: {transform_info['shift']:.4f}")
            else:  # Log 변환
                if transform_info['shift'] > 0:
                    st.write(f"- 이동값: {transform_info['shift']:.4f}")
        
        with col2:
            st.write("**변환 상세:**")
            st.write(f"- 변환 유형: {transform_info['family']}")
            st.write(f"- 변환 공식: {transform_info['formula']}")
            st.write(f"- 원본 데이터 R²: {r_orig**2:.4f}")
            st.write(f"- 변환 데이터 R²: {r_trans**2:.4f}")
    
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
    
    2. **Anderson-Darling 검정**
       - 표본 크기가 클 때도 사용 가능
       - 꼬리 부분의 차이에 더 민감
    
    #### 데이터 변환 방법
    데이터가 정규분포를 따르지 않을 경우, 다음과 같은 변환 방법을 사용할 수 있습니다:
    
    1. **Johnson 변환**
       - 다양한 형태의 비정규 분포를 정규분포로 변환
       - 가장 강력한 변환 방법 중 하나
    
    2. **Box-Cox 변환**
       - 데이터의 형태에 따라 자동으로 최적의 변환 방법 선택
       - 양수 데이터에만 적용 가능
    
    3. **Log 변환**
       - 오른쪽으로 치우친 분포를 정규분포에 가깝게 변환
       - 양수 데이터에만 적용 가능
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
            
            # 정규성 검정 결과 표시
            st.subheader("정규성 검정 결과")
            
            if shapiro_p < 0.05:
                st.warning(f"⚠️ Shapiro-Wilk 검정 결과: 데이터가 정규분포를 따르지 않습니다. (p = {shapiro_p:.4f})")
                
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
                        
                        # 변환된 데이터의 정규성 검정
                        transformed_stat, transformed_p = stats.shapiro(transformed_data)
                        
                        if transformed_p >= 0.05:
                            st.success(f"✅ 변환 후 Shapiro-Wilk 검정 결과: 데이터가 정규분포를 따릅니다. (p = {transformed_p:.4f})")
                        else:
                            st.warning(f"⚠️ 변환 후에도 정규성을 만족하지 않습니다. (p = {transformed_p:.4f})")
                        
                        # 변환 데이터 사용 여부 선택
                        use_transformed = st.checkbox(
                            "이 변환된 데이터를 이후 분석에 사용하시겠습니까?",
                            help="체크하면 변환된 데이터가 공정능력분석 등에 사용됩니다."
                        )
                        
                        if use_transformed:
                            # 변환 정보 저장
                            st.session_state.transformed_vars[selected_var] = {
                                'data': transformed_data,
                                'method': transform_info['family'],
                                'info': transform_info,
                                'r_squared': transformed_p,
                                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'original_data': var_data.copy()
                            }
                            
                            # 변환된 데이터를 원본 데이터프레임에 저장
                            if 'transformed_data' not in st.session_state:
                                st.session_state.transformed_data = st.session_state.data.copy()
                            st.session_state.transformed_data[selected_var] = transformed_data
                            
                            st.success(f"""
                            ✅ 변환된 데이터가 저장되었습니다.
                            - 변수: {selected_var}
                            - 변환 방법: {transform_info['family']}
                            - R² 값: {transformed_p:.4f}
                            """)
                        else:
                            # 변환 정보 제거
                            if selected_var in st.session_state.transformed_vars:
                                del st.session_state.transformed_vars[selected_var]
                            st.info("원본 데이터를 사용합니다.")
                else:
                    # 변환하지 않음 선택 시 변환 정보 제거
                    if selected_var in st.session_state.transformed_vars:
                        del st.session_state.transformed_vars[selected_var]
                    st.info("원본 데이터를 사용합니다.")
            else:
                st.success(f"✅ Shapiro-Wilk 검정 결과: 데이터가 정규분포를 따릅니다. (p = {shapiro_p:.4f})")
                # 정규성을 만족하는 경우 변환 정보 제거
                if selected_var in st.session_state.transformed_vars:
                    del st.session_state.transformed_vars[selected_var]
            
            # 현재 변환 상태 표시
            if st.session_state.transformed_vars:
                st.subheader("현재 적용된 변환 목록")
                for var, info in st.session_state.transformed_vars.items():
                    st.write(f"""
                    - 변수: {var}
                    - 변환 방법: {info['method']}
                    - R² 값: {info['r_squared']:.4f}
                    - 변환 날짜: {info['timestamp']}
                    """)
        else:
            st.error(f"선택한 변수 '{selected_var}'에 유효한 데이터가 없습니다.")
else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")

# 페이지 하단 소개
st.markdown("---")
st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ") 