import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프를 중앙에 표시하는 헬퍼 함수 추가
def display_plot_centered(fig, width_pct=90):
    """그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.pyplot(fig)

# Plotly 그래프를 중앙에 표시하는 헬퍼 함수 추가
def display_plotly_centered(fig, width_pct=90):
    """Plotly 그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=True)

# 이상치 탐지 함수 추가
def detect_outliers(data, method='IQR', threshold=1.5):
    """이상치를 탐지하는 함수
    
    Parameters:
    -----------
    data : pandas.Series
        이상치를 탐지할 데이터
    method : str, default 'IQR'
        이상치 탐지 방법, 'IQR' 또는 'Z-Score'
    threshold : float, default 1.5
        IQR 방법에서는 1.5 (일반적) 또는 3.0 (극단값만), Z-Score 방법에서는 3.0이 일반적
        
    Returns:
    --------
    pandas.Series
        이상치 여부를 나타내는 불리언 시리즈 (True: 이상치)
    """
    if method == 'IQR':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == 'Z-Score':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    else:
        raise ValueError("method 인자는 'IQR' 또는 'Z-Score'여야 합니다.")

st.set_page_config(
    page_title="공정능력분석",
    page_icon="📊",
    layout="wide"
)

st.title("1. 공정능력분석")

# 공정능력분석 설명 추가
with st.expander("📚 공정능력분석이란?"):
    st.markdown("""
    ### 공정능력분석(Process Capability Analysis)
    
    공정능력분석은 생산 공정이 고객 요구사항이나 제품 규격을 충족시킬 수 있는 능력을 통계적으로 평가하는 방법입니다.
    
    ### 주요 지표 (정규분포 가정 시)
    
    - **Cp (공정능력지수)**: 공정의 산포와 규격 폭의 비율
      - Cp = (USL - LSL) / (6σ)
      - Cp ≥ 1.33: 우수, 1.00 ≤ Cp < 1.33: 적절, Cp < 1.00: 부적합
    
    - **Cpk (공정능력지수K)**: 공정의 산포와 중심이탈을 함께 고려
      - Cpk = min[(USL - μ) / (3σ), (μ - LSL) / (3σ)]
      - Cpk ≥ 1.33: 우수, 1.00 ≤ Cpk < 1.33: 적절, Cpk < 1.00: 부적합
    
    - **Cpu (상한 공정능력지수)**: 상한규격에 대한 공정능력
      - Cpu = (USL - μ) / (3σ)
    
    - **Cpl (하한 공정능력지수)**: 하한규격에 대한 공정능력
      - Cpl = (μ - LSL) / (3σ)
      
    ### 비모수적 지표 (정규분포 가정이 성립하지 않을 때)
    
    - **Pp (백분위수 기반 공정능력지수)**: 
      - Pp = (USL - LSL) / (P99.865 - P0.135)
      - 여기서 P99.865와 P0.135는 각각 99.865% 및 0.135% 백분위수
    
    - **Ppk (백분위수 기반 공정능력지수K)**: 
      - Ppk = min[(USL - P50) / (P99.865 - P50), (P50 - LSL) / (P50 - P0.135)]
      - 여기서 P50은 중앙값(50% 백분위수)
    """)

# 데이터 확인
if 'data' in st.session_state and st.session_state.data is not None:
    data = st.session_state.data
    
    # 변수 선택
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_var = st.selectbox(
            "분석할 변수 선택:",
            options=numeric_cols
        )
        
        # 데이터 기본값 계산
        var_data_original = data[selected_var].dropna()
        
        # 이상치 처리 옵션 추가
        st.subheader("이상치 처리 옵션")
        
        # 이상치 처리에 대한 설명 추가
        with st.expander("📚 이상치란? 이상치 처리가 왜 중요한가요?"):
            st.markdown("""
            ### 이상치(Outlier)란?
            
            이상치는 다른 관측값들과 동떨어진, 비정상적으로 큰 값이나 작은 값을 의미합니다. 
            이상치는 실제 공정의 문제, 측정 오류, 또는 데이터 입력 오류 등 다양한 원인으로 발생할 수 있습니다.
            
            ### 이상치가 공정능력분석에 미치는 영향
            
            이상치는 다음과 같은 문제를 일으킬 수 있습니다:
            
            1. **평균 및 표준편차 왜곡**: 이상치는 데이터의 평균과 표준편차를 크게 왜곡시킬 수 있습니다.
            2. **공정능력지수 과소평가**: 이상치로 인해 표준편차가 증가하면 Cp, Cpk 등의 공정능력지수가 실제보다 낮게 계산될 수 있습니다.
            3. **공정 안정성 오판**: 이상치를 포함한 분석은 안정적인 공정을 불안정하다고 잘못 판단하게 할 수 있습니다.
            
            ### 이상치 탐지 방법
            
            #### 1. IQR(Interquartile Range) 방법
            - **원리**: 데이터의 1사분위수(Q1)와 3사분위수(Q3) 사이의 거리(IQR)를 기반으로 함
            - **이상치 판단**: Q1 - k×IQR 보다 작거나 Q3 + k×IQR 보다 큰 값 (k는 보통 1.5 또는 3)
            - **장점**: 데이터 분포에 덜 민감하며, 비대칭 분포에서도 비교적 잘 작동함
            - **적합한 상황**: 데이터가 정규분포가 아니거나, 분포 형태를 잘 모를 때
            
            #### 2. Z-Score 방법
            - **원리**: 각 데이터 포인트가 평균으로부터 얼마나 떨어져 있는지를 표준편차 단위로 측정
            - **이상치 판단**: |Z| > k (보통 k=3, 즉 평균에서 3 표준편차 이상 떨어진 값)
            - **장점**: 직관적이고 계산이 간단함
            - **적합한 상황**: 데이터가 대략 정규분포를 따를 때
            
            ### 이상치 처리 방법 선택 시 고려사항
            
            - **제거**: 이상치가 측정 오류나 데이터 입력 오류로 확인된 경우 적합
            - **시각적으로 표시만**: 이상치가 실제 공정의 이상을 나타낼 수 있는 경우, 제거하지 않고 표시만 하여 추가 조사 가능
            - **대체**: (현재 이 도구에서는 지원하지 않음) 이상치를 중앙값이나 평균 등으로 대체하는 방법
            
            ### 주의사항
            
            - 모든 이상치가 오류는 아닙니다. 일부 이상치는 중요한 정보를 제공할 수 있습니다.
            - 이상치 처리는 데이터의 특성과 업무 맥락을 고려하여 신중하게 수행해야 합니다.
            - 이상치 처리 전/후의 결과를 비교하여 처리의 영향을 평가하는 것이 좋습니다.
            """)

        outlier_col1, outlier_col2 = st.columns(2)

        with outlier_col1:
            use_outlier_treatment = st.checkbox("이상치 처리 활성화", value=False, 
                                             help="데이터에서 이상치를 탐지하고 처리합니다.")

        if use_outlier_treatment:
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
            outliers = detect_outliers(var_data_original, method=outlier_method, threshold=threshold)
            outlier_count = outliers.sum()
            
            # 이상치 처리 방법 선택
            outlier_treatment = st.radio(
                "이상치 처리 방법",
                options=["제거", "시각적으로 표시만"],
                index=0,
                help="이상치를 제거하거나 시각적으로만 표시할 수 있습니다."
            )
            
            # 이상치 정보 표시
            if outlier_count > 0:
                st.info(f"탐지된 이상치: {outlier_count}개 ({outlier_count/len(var_data_original):.1%})")
                
                # 이상치 데이터 표시
                if st.checkbox("이상치 데이터 보기"):
                    st.dataframe(data[outliers][selected_var])
                    
                    if outlier_treatment == "제거":
                        st.caption("⚠️ 위 이상치들은 분석에서 제외됩니다.")
                    else:
                        st.caption("ℹ️ 위 이상치들은 그래프에 표시되며 분석에 포함됩니다.")
            else:
                st.success("이상치가 탐지되지 않았습니다.")
            
            # 이상치 처리
            if outlier_treatment == "제거" and outlier_count > 0:
                var_data = var_data_original[~outliers].copy()
                st.warning(f"이상치 {outlier_count}개가 제거되었습니다. 남은 데이터: {len(var_data)}개")
            else:
                var_data = var_data_original.copy()
        else:
            var_data = var_data_original.copy()
        
        # 규격 한계 설정
        st.subheader("규격 한계 설정")
        
        # 데이터 통계량 계산
        mean_val = var_data.mean()
        std_val = var_data.std()
        min_val = var_data.min()
        max_val = var_data.max()
        
        # 자동 규격 제안
        suggested_lsl = mean_val - 3*std_val
        suggested_usl = mean_val + 3*std_val
        
        col1, col2 = st.columns(2)
        with col1:
            lsl = st.number_input("하한규격(LSL):", 
                                value=float(suggested_lsl), 
                                format="%.2f",
                                help="제품 규격의 하한값 (기본값: 평균-3σ)")
        with col2:
            usl = st.number_input("상한규격(USL):", 
                                value=float(suggested_usl), 
                                format="%.2f",
                                help="제품 규격의 상한값 (기본값: 평균+3σ)")
        
        if len(var_data) > 0:
            # 정규성 검정
            try:
                if len(var_data) < 8:
                    st.warning(f"정규성 검정을 수행하기 위해서는 최소 8개 이상의 데이터가 필요합니다. 현재 데이터 개수: {len(var_data)}개")
                    normality_result = "데이터 부족으로 검정 불가"
                    shapiro_result = None
                    k2_result = None
                else:
                    # Shapiro-Wilk 검정 (주요 검정으로 사용)
                    shapiro_stat, shapiro_p = stats.shapiro(var_data)
                    
                    # D'Agostino-Pearson 검정 (보조 검정으로 사용)
                    k2, p_value = stats.normaltest(var_data)
                    
                    # 결과 해석 (Shapiro-Wilk 기준)
                    if shapiro_p < 0.05:
                        normality_result = "비정규 분포 (p < 0.05)"
                    else:
                        normality_result = "정규 분포 (p >= 0.05)"
                    
                    shapiro_result = f"Shapiro-Wilk 검정: W = {shapiro_stat:.3f}, p-value = {shapiro_p:.4f}"
                    k2_result = f"D'Agostino-Pearson 검정: k² = {k2:.3f}, p-value = {p_value:.4f}"
            except Exception as e:
                st.error(f"정규성 검정 중 오류가 발생했습니다: {str(e)}")
                normality_result = "검정 오류"
                shapiro_result = None
                k2_result = None
            
            # 공정능력 지수 계산
            if std_val > 0:
                # 정규성을 만족하는 경우의 공정능력지수
                if normality_result == "정규 분포 (p >= 0.05)":
                    cp = (usl - lsl) / (6 * std_val)
                    cpu = (usl - mean_val) / (3 * std_val)
                    cpl = (mean_val - lsl) / (3 * std_val)
                    cpk = min(cpu, cpl)
                    
                    # 계산 방법 표시
                    method_used = "정규분포 가정"
                    
                # 정규성을 만족하지 않는 경우의 비모수적 공정능력지수
                else:
                    # 백분위수 계산
                    p99865 = np.percentile(var_data, 99.865)
                    p00135 = np.percentile(var_data, 0.135)
                    p50 = np.percentile(var_data, 50)  # 중앙값
                    
                    # 비모수적 공정능력지수 계산
                    pp = (usl - lsl) / (p99865 - p00135)
                    ppu = (usl - p50) / (p99865 - p50)
                    ppl = (p50 - lsl) / (p50 - p00135)
                    ppk = min(ppu, ppl)
                    
                    # 기존 변수에 매핑하여 기존 코드와 호환성 유지
                    cp = pp
                    cpu = ppu
                    cpl = ppl
                    cpk = ppk
                    
                    # 계산 방법 표시
                    method_used = "비모수적 방법(백분위수 기반)"
            else:
                st.warning("표준편차가 0입니다. 공정능력지수를 계산할 수 없습니다.")
                cp = np.nan
                cpu = np.nan
                cpl = np.nan
                cpk = np.nan
                method_used = "계산 불가"
            
            # 공정관리도 (Run Chart) - Plotly 사용
            # 인덱스가 실제 행 이름인지 확인
            if isinstance(var_data.index, pd.RangeIndex):
                # 기본 숫자 인덱스인 경우
                x_values = list(range(len(var_data)))
                hover_text = [f"관측치: {i+1}<br>값: {v:.2f}" for i, v in enumerate(var_data)]
            else:
                # 의미 있는 인덱스인 경우
                x_values = list(range(len(var_data)))
                hover_text = [f"ID: {idx}<br>값: {v:.2f}" for idx, v in zip(var_data.index, var_data)]
            
            # Plotly 인터랙티브 차트 생성
            fig_plotly = go.Figure()
            
            # 데이터 라인 추가
            fig_plotly.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=var_data.values,
                    mode='lines+markers',
                    name=selected_var,
                    line=dict(color='blue'),
                    marker=dict(size=6),
                    text=hover_text,
                    hoverinfo='text'
                )
            )
            
            # 이상치 데이터 표시 (사용자가 '시각적으로 표시만' 선택 시)
            if use_outlier_treatment and outlier_treatment == "시각적으로 표시만" and outlier_count > 0:
                # 이상치 데이터 포인트만 가져오기
                outlier_data = var_data_original[outliers]
                
                # 이상치의 인덱스를 x_values에 매핑
                outlier_indices = []
                for idx in outlier_data.index:
                    try:
                        # 원본 데이터에서 이상치 인덱스 찾기
                        pos = var_data_original.index.get_loc(idx)
                        outlier_indices.append(pos)
                    except:
                        continue
                
                # 이상치를 빨간색 X로 표시
                if outlier_indices:
                    outlier_y = [var_data_original.iloc[i] for i in outlier_indices]
                    outlier_hover = [f"이상치 ID: {var_data_original.index[i]}<br>값: {var_data_original.iloc[i]:.2f}" for i in outlier_indices]
                    
                    fig_plotly.add_trace(
                        go.Scatter(
                            x=[outlier_indices], 
                            y=[outlier_y],
                            mode='markers',
                            name='이상치',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='x'
                            ),
                            text=outlier_hover,
                            hoverinfo='text'
                        )
                    )
            
            # 기준선 추가
            fig_plotly.add_trace(go.Scatter(x=x_values, y=[mean_val]*len(x_values), mode='lines', name='평균', line=dict(color='green', width=2)))
            fig_plotly.add_trace(go.Scatter(x=x_values, y=[mean_val + 3*std_val]*len(x_values), mode='lines', name='+3σ', line=dict(color='red', dash='dash')))
            fig_plotly.add_trace(go.Scatter(x=x_values, y=[mean_val - 3*std_val]*len(x_values), mode='lines', name='-3σ', line=dict(color='red', dash='dash')))
            fig_plotly.add_trace(go.Scatter(x=x_values, y=[usl]*len(x_values), mode='lines', name='USL', line=dict(color='purple', dash='dashdot')))
            fig_plotly.add_trace(go.Scatter(x=x_values, y=[lsl]*len(x_values), mode='lines', name='LSL', line=dict(color='purple', dash='dashdot')))
            
            # X축 레이블 설정
            if not isinstance(var_data.index, pd.RangeIndex):
                fig_plotly.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=x_values,
                        ticktext=var_data.index,
                        tickangle=0
                    )
                )
            
            # 그래프 레이아웃 설정
            fig_plotly.update_layout(
                title=f'{selected_var} 공정관리도',
                xaxis_title='관측치',
                yaxis_title='값',
                hovermode='closest',
                height=500,
                width=900,
                margin=dict(l=50, r=30, t=50, b=50)
            )
            
            # 그리드 추가
            fig_plotly.update_layout(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            )
            
            # 그래프 표시
            st.subheader("공정관리도 (인터랙티브)")
            st.caption("👉 각 점에 마우스를 올리면 자세한 정보를 볼 수 있습니다")
            display_plotly_centered(fig_plotly)
            
            # 이상치 처리 정보 표시
            if use_outlier_treatment:
                st.caption(f"📊 이상치 처리: {outlier_method} 방법, 임계값 {threshold}, 처리 방법: {outlier_treatment}")
                if outlier_treatment == "제거" and outlier_count > 0:
                    st.caption(f"🔍 이상치 {outlier_count}개 제거 후 분석 수행, 남은 데이터: {len(var_data)}개")
            
            # 공정능력 지수 표시
            st.subheader('공정능력 분석 결과')
            
            # 정규성 검정 결과 표시
            if normality_result == "정규 분포 (p >= 0.05)":
                st.success(f"✅ 정규성 검정 결과: 정규분포 가정을 만족합니다 ({shapiro_result})")
            else:
                st.warning(f"⚠️ 정규성 검정 결과: {normality_result} ({shapiro_result})")
                st.info("🔍 비모수적 방법(백분위수 기반)을 사용하여 공정능력지수를 계산합니다.")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                # 공정능력지수 표시
                cp_display = f"{cp:.2f}" if not np.isnan(cp) else "N/A"
                cp_name = "Cp" if normality_result == "정규 분포 (p >= 0.05)" else "Pp"
                st.metric(cp_name, cp_display, 
                         delta="주의 필요" if not np.isnan(cp) and cp >= 1 and cp < 1.33 else
                               "적합" if not np.isnan(cp) and cp >= 1.33 else
                               "부적합" if not np.isnan(cp) and cp < 1 else "계산 불가")
                st.caption("공정의 산포가 규격 대비 얼마나 좁은지")
            
            with metrics_col2:
                cpk_display = f"{cpk:.2f}" if not np.isnan(cpk) else "N/A"
                cpk_name = "Cpk" if normality_result == "정규 분포 (p >= 0.05)" else "Ppk"
                st.metric(cpk_name, cpk_display, 
                         delta="주의 필요" if not np.isnan(cpk) and cpk >= 1 and cpk < 1.33 else
                               "적합" if not np.isnan(cpk) and cpk >= 1.33 else
                               "부적합" if not np.isnan(cpk) and cpk < 1 else "계산 불가")
                st.caption("공정 산포와 중심위치를 모두 고려한 지수")
            
            with metrics_col3:
                cpu_display = f"{cpu:.2f}" if not np.isnan(cpu) else "N/A"
                cpu_name = "Cpu" if normality_result == "정규 분포 (p >= 0.05)" else "Ppu"
                st.metric(cpu_name, cpu_display)
                st.caption("상한규격 기준 공정능력")
            
            with metrics_col4:
                cpl_display = f"{cpl:.2f}" if not np.isnan(cpl) else "N/A"
                cpl_name = "Cpl" if normality_result == "정규 분포 (p >= 0.05)" else "Ppl"
                st.metric(cpl_name, cpl_display)
                st.caption("하한규격 기준 공정능력")
            
            # 통계 요약 테이블
            st.subheader('통계 요약')
            
            stats_df = pd.DataFrame({
                '통계량': ['평균', '표준편차', '중앙값', '최소값', '최대값', 'LSL', 'USL', 
                        f'{cp_name}', f'{cpk_name}', '계산 방법'],
                '값': [f'{mean_val:.2f}', f'{std_val:.2f}', f'{np.median(var_data):.2f}', 
                      f'{min_val:.2f}', f'{max_val:.2f}', f'{lsl:.2f}', f'{usl:.2f}', 
                      cp_display, cpk_display, method_used]
            })
            
            st.table(stats_df)
            
            # 공정능력 해석
            st.subheader('공정능력 판정')
            
            # 공정능력 해석을 표로 정리
            interpretation_df = pd.DataFrame(columns=["지표", "값", "판정", "개선 방향"])
            
            # Cp/Pp 해석
            cp_name = "Cp" if normality_result == "정규 분포 (p >= 0.05)" else "Pp"
            if np.isnan(cp):
                cp_judgment = "계산 불가"
                cp_action = "데이터 확인 필요"
            elif cp >= 1.33:
                cp_judgment = "우수함"
                cp_action = "현상 유지"
            elif cp >= 1.0:
                cp_judgment = "적절함"
                cp_action = "지속적 개선 필요"
            else:
                cp_judgment = "부적합"
                cp_action = "공정 산포 감소 필요"
                
            # Cpk/Ppk 해석
            cpk_name = "Cpk" if normality_result == "정규 분포 (p >= 0.05)" else "Ppk"
            if np.isnan(cpk):
                cpk_judgment = "계산 불가"
                cpk_action = "데이터 확인 필요"
            elif cpk >= 1.33:
                cpk_judgment = "우수함"
                cpk_action = "현상 유지"
            elif cpk >= 1.0:
                cpk_judgment = "적절함"
                cpk_action = "중심 조정 또는 산포 감소 필요"
            else:
                cpk_judgment = "부적합"
                cpk_action = "공정 중심 조정 및 산포 감소 시급"
                
            # 중심 치우침 해석
            central_value = mean_val if normality_result == "정규 분포 (p >= 0.05)" else np.median(var_data)
            if abs(central_value - (usl + lsl) / 2) > std_val:
                center_judgment = "치우침 있음"
                center_action = "공정 중심 조정 필요"
            else:
                center_judgment = "양호함"
                center_action = "현상 유지"
                
            # 정규성 해석
            if normality_result == "정규 분포 (p >= 0.05)":
                normal_judgment = "정규 분포"
                normal_action = "표준 공정능력분석 적용 가능"
            else:
                normal_judgment = "비정규 분포"
                normal_action = "비모수적 방법 사용 중"
            
            # 데이터프레임에 추가
            interpretation_df.loc[0] = [f"공정능력({cp_name})", cp_display, cp_judgment, cp_action]
            interpretation_df.loc[1] = [f"공정능력지수K({cpk_name})", cpk_display, cpk_judgment, cpk_action]
            interpretation_df.loc[2] = ["공정 중심", f"{central_value:.2f}", center_judgment, center_action]
            interpretation_df.loc[3] = ["정규성", f"{shapiro_result}", normal_judgment, normal_action]
            
            st.table(interpretation_df)
            
            # 종합 해석
            if not np.isnan(cp) and not np.isnan(cpk):
                if cp >= 1.33 and cpk >= 1.33:
                    st.success('✅ 종합 판정: 공정이 안정적이며 규격에 대한 여유도가 충분합니다.')
                elif cp >= 1.0 and cpk >= 1.0:
                    st.warning('⚠️ 종합 판정: 공정이 규격을 만족하나, 개선의 여지가 있습니다.')
                else:
                    st.error('❌ 종합 판정: 공정이 불안정하거나 규격을 벗어날 위험이 있습니다. 개선이 필요합니다.')
            else:
                st.error('❌ 종합 판정: 공정능력 지수 계산에 문제가 있습니다. 데이터와 규격을 확인하세요.')
            
            # 실용적인 조언 추가
            st.subheader("💡 개선 방안")
            
            # 정규성에 따른 추가 설명
            if normality_result != "정규 분포 (p >= 0.05)":
                st.info("""
                📌 **비정규 분포 데이터에 대한 참고 사항**:
                - 백분위수 기반 계산법(Pp, Ppk)이 사용되었습니다.
                - 정규성을 가정한 지표(Cp, Cpk)보다 더 보수적인 평가일 수 있습니다.
                - 데이터 변환(로그, 제곱근 등)을 통해 정규성을 개선할 수 있는지 검토해보세요.
                """)
            
            if not np.isnan(cp) and not np.isnan(cpk):
                if cp < cpk:
                    st.info("이론적으로 Cp는 항상 Cpk보다 크거나 같아야 합니다. 데이터를 재확인하세요.")
                elif cp > cpk:
                    # 정규성에 따라 다른 메시지 표시
                    central_term = "평균" if normality_result == "정규 분포 (p >= 0.05)" else "중앙값"
                    st.info(f"공정 중심({central_term})을 규격 중심({(usl+lsl)/2:.2f})에 맞추면 {cpk_name}를 {cp:.2f}까지 향상시킬 수 있습니다.")
                
                if not np.isnan(cpk) and cpk < 1.0:
                    if not np.isnan(cpu) and not np.isnan(cpl):
                        if cpu < cpl:
                            st.info(f"공정 {central_term}을 낮추면 {cpk_name}를 개선할 수 있습니다.")
                        elif cpl < cpu:
                            st.info(f"공정 {central_term}을 높이면 {cpk_name}를 개선할 수 있습니다.")
            
            # 공정능력 시각적 해석 (시각화 도움말)
            with st.expander("📊 그래프 해석 방법"):
                st.markdown(f"""
                ### 공정관리도 해석
                - **빨간 점선(±3σ)**: 관리 한계선으로, 이 범위를 벗어나면 공정이 불안정할 수 있습니다.
                - **초록 실선(평균)**: 공정의 중심을 나타냅니다.
                - **보라색 점선(USL/LSL)**: 제품 규격 한계를 나타냅니다.
                
                ### 히스토그램 해석
                - **정규분포 여부**: p값이 0.05 이상이면 정규분포로 간주합니다(현재 p={shapiro_result}).
                - **종 모양에 가까울수록**: 정규분포를 따르는 안정적인 공정입니다.
                - **규격선(USL/LSL)이 분포 바깥에 있을수록**: 공정능력이 우수합니다.
                - **규격선이 분포 안에 있다면**: 불량품 발생 가능성이 있습니다.
                
                ### {cp_name}와 {cpk_name}의 차이
                - **{cp_name}**: 이상적인 공정 능력(산포만 고려)
                - **{cpk_name}**: 실제 공정 능력(산포와 중심 모두 고려)
                """)
                
                # 정규성에 따른 추가 설명
                if normality_result != "정규 분포 (p >= 0.05)":
                    st.markdown("""
                    ### 비모수적 방법(백분위수)에 대한 추가 설명
                    - **Pp**: 99.865% 및 0.135% 백분위수 간의 거리로 계산됩니다.
                    - **Ppk**: 중앙값과 99.865% 또는 0.135% 백분위수 사이의 거리 중 작은 값으로 계산됩니다.
                    - 이 방식은 데이터가 정규분포를 따르지 않을 때 더 정확한 공정능력을 평가합니다.
                    """)

            # 비모수적 방법에 대한 추가 설명을 여기에 넣으세요
            if normality_result != "정규 분포 (p >= 0.05)":
                with st.expander("🔍 비모수적 공정능력지수(Pp, Ppk) 쉽게 이해하기"):
                    st.markdown("""
                    ### 비모수적 공정능력지수 쉽게 이해하기
                    
                    #### 왜 Pp와 Ppk가 필요한가요?
                    - 많은 실제 공정 데이터는 종 모양의 정규분포를 따르지 않습니다.
                    - 데이터가 정규분포가 아닐 때 기존 Cp, Cpk를 사용하면 **잘못된 결론**을 내릴 수 있습니다.
                    - Pp와 Ppk는 데이터의 분포 형태에 상관없이 사용할 수 있는 **더 신뢰할 수 있는 지표**입니다.
                    
                    #### 쉽게 설명하자면...
                    - **Cp/Cpk**: "데이터가 종 모양이라고 가정하고" 공정 능력을 평가
                    - **Pp/Ppk**: "데이터의 실제 모양을 그대로 반영해서" 공정 능력을 평가
                    
                    #### 실제 예시로 이해하기
                    마치 키 180cm인 사람을 기준으로 만든 옷을 모든 사람에게 맞춰보는 것(Cp/Cpk)과, 
                    각 사람의 실제 치수를 측정해서 맞춤 옷을 만드는 것(Pp/Ppk)의 차이라고 볼 수 있습니다.
                    
                    #### 계산 방식의 차이
                    - **Cp**: 표준편차(σ)를 사용 → 정규분포 가정에 의존
                    - **Pp**: 백분위수(99.865%와 0.135% 사이 간격)를 사용 → 실제 데이터 분포 반영
                    
                    #### 간단히 말하면
                    - **Pp** = 규격 폭 ÷ 데이터의 실제 퍼짐 정도
                    - **Ppk** = 규격 한계선과 데이터 중심(중앙값) 사이의 가장 가까운 거리 ÷ 데이터의 한쪽 퍼짐 정도
                    
                    #### 판단 기준은 동일합니다
                    - Pp, Ppk ≥ 1.33: 우수한 공정
                    - 1.00 ≤ Pp, Ppk < 1.33: 적절한 공정
                    - Pp, Ppk < 1.00: 개선이 필요한 공정
                    """)
                    
                    # 시각적 설명을 위한 간단한 다이어그램 (선택적)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.linspace(-4, 4, 1000)
                    y1 = stats.norm.pdf(x, 0, 1)  # 정규분포
                    y2 = stats.skewnorm.pdf(x, 5, 0, 1.5)  # 비대칭분포
                    
                    ax.plot(x, y1, 'b-', label='정규분포 (Cp/Cpk 적합)')
                    ax.plot(x, y2, 'r-', label='비대칭분포 (Pp/Ppk 필요)')
                    
                    # 정규분포의 ±3σ 지점
                    ax.axvline(x=-3, color='blue', linestyle='--', alpha=0.5)
                    ax.axvline(x=3, color='blue', linestyle='--', alpha=0.5)
                    
                    # 비대칭분포의 0.135% 및 99.865% 백분위수 지점
                    p_low = stats.skewnorm.ppf(0.00135, 5, 0, 1.5)
                    p_high = stats.skewnorm.ppf(0.99865, 5, 0, 1.5)
                    ax.axvline(x=p_low, color='red', linestyle='--', alpha=0.5)
                    ax.axvline(x=p_high, color='red', linestyle='--', alpha=0.5)
                    
                    ax.set_title('정규분포와 비대칭분포 비교')
                    ax.set_xlabel('값')
                    ax.set_ylabel('밀도')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
        else:
            st.error(f"선택한 변수 '{selected_var}'에 유효한 데이터가 없습니다.")
    else:
        st.error("분석할 숫자형 변수가 없습니다.")
else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")