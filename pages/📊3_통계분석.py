import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io # io 모듈 추가

# 한글 폰트 설정 (matplotlib용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프를 중앙에 표시하는 헬퍼 함수 추가
def display_plotly_centered(fig, width_pct=60):
    """Plotly 그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=True)

st.title("2. 통계분석")

# 통계분석 개념 설명 추가
with st.expander("📚 통계분석이란?"):
    st.markdown("""
    ### 통계분석(Statistical Analysis)
    통계분석은 데이터에서 의미 있는 패턴과 관계를 발견하기 위한 방법입니다. 이 페이지에서는 다음과 같은 분석을 제공합니다:
    
    ### 상관관계 분석
    **상관계수**는 두 변수 간의 선형 관계 강도를 -1에서 1 사이의 값으로 나타냅니다:
    - **+1에 가까울수록**: 강한 양의 상관관계 (한 변수가 증가하면 다른 변수도 증가)
    - **-1에 가까울수록**: 강한 음의 상관관계 (한 변수가 증가하면 다른 변수는 감소)
    - **0에 가까울수록**: 상관관계가 약함 (두 변수가 독립적)
    
    **해석 지침**:
    - |r| > 0.7: 강한 상관관계
    - 0.3 < |r| < 0.7: 중간 정도의 상관관계
    - |r| < 0.3: 약한 상관관계
    
    ### 산점도(Scatter Plot)
    두 변수 간의 관계를 시각적으로 표현한 그래프입니다:
    - **점들이 직선 형태로 모여있을수록**: 선형 관계가 강함
    - **점들이 넓게 퍼져있을수록**: 상관관계가 약함
    - **빨간 선**: 회귀선으로, 데이터의 전반적인 추세를 보여줍니다
    
    ### 박스플롯(Box Plot)
    데이터의 분포와 이상치를 시각화하는 그래프입니다:
    - **박스**: 1사분위수(25%)에서 3사분위수(75%)까지의 범위 (IQR)
    - **중앙선**: 중앙값(median)
    - **수염**: 정상 범위의 최소/최대값
    - **점**: 이상치(outlier)
    
    ### 기술통계량
    데이터의 주요 특성을 요약한 수치들:
    - **count**: 데이터 개수
    - **mean**: 평균
    - **std**: 표준편차
    - **min/max**: 최소/최대값
    - **25%/50%/75%**: 1사분위수/중앙값/3사분위수
    
    ### 범주형 데이터 빈도 분석
    범주형 데이터의 분포를 분석하는 방법입니다:
    - **빈도(Frequency)**: 각 범주별 데이터 개수
    - **비율(Percentage)**: 전체 데이터 중 각 범주가 차지하는 비율
    - **시각화**: 막대 그래프를 통해 각 범주의 빈도를 직관적으로 비교
    - **활용**: 제품 유형, 공급업체, 라인 등 범주형 변수의 분포 파악에 유용
    
    ### 집단 간 비교 분석
    서로 다른 그룹 간의 차이를 통계적으로 검증하는 방법입니다:
    
    **1. 두 그룹 비교 (T-검정/Mann-Whitney U 검정)**
    - **정규성 검증**: Shapiro-Wilk 검정으로 데이터의 정규성 확인
    - **모수 검정(T-검정)**: 정규성을 만족하는 경우 사용
    - **비모수 검정(Mann-Whitney U)**: 정규성을 만족하지 않는 경우 사용
    - **해석**: p-value < 0.05인 경우 두 그룹 간 차이가 통계적으로 유의미함
    
    **2. 세 그룹 이상 비교 (ANOVA/Kruskal-Wallis H 검정)**
    - **정규성 검증**: 각 그룹별 정규성 확인
    - **모수 검정(ANOVA)**: 모든 그룹이 정규성을 만족하는 경우 사용
    - **비모수 검정(Kruskal-Wallis H)**: 하나 이상의 그룹이 정규성을 만족하지 않는 경우 사용
    - **해석**: p-value < 0.05인 경우 그룹 간 차이가 통계적으로 유의미함
    """)

# 데이터 확인
if 'data' in st.session_state and st.session_state.data is not None:
    data = st.session_state.data
    
    # 상관관계 분석 섹션
    st.subheader("상관관계 분석")
    
    # 변수 선택 (숫자형 데이터만)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    target_var = st.selectbox("기준 변수 선택:", numeric_cols)
    
    if target_var:
        # 상관관계 계산
        corr_data = data[numeric_cols].corr()[target_var].sort_values(ascending=False)
        corr_data = corr_data.drop(target_var)  # 자기 자신 제외
        
        # 절대값으로 상관계수가 높은 상위 변수들을 선택 (정렬 기준)
        top_corrs_indices = corr_data.abs().sort_values(ascending=False).head(10).index
        
        # 선택된 변수들의 실제 상관계수 값 (절대값 아님)
        top_corrs = corr_data[top_corrs_indices]
        
        # Plotly를 사용한 상관관계 시각화
        fig_corr = go.Figure()
        
        # 상관계수에 따라 색상 설정
        colors = ['#3498db' if val > 0 else '#e74c3c' for val in top_corrs.values]
        
        # 가로 막대 차트 추가
        fig_corr.add_trace(
            go.Bar(
                y=top_corrs.index,
                x=top_corrs.values,  # 원래 값 사용 (양수/음수 유지)
                orientation='h',
                marker_color=colors,
                text=[f'{val:.2f}' for val in top_corrs.values],
                textposition='outside',
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            )
        )
        
        # 레이아웃 설정
        fig_corr.update_layout(
            title=f'{target_var}와(과)의 상관관계 (상위 10개)',
            xaxis_title='상관계수',
            yaxis_title='변수',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                range=[-1, 1],  # x축 범위를 -1에서 1로 고정
                zeroline=True,  # 0 기준선 추가
                zerolinecolor='black',
                zerolinewidth=1
            )
        )
        
        # 중앙에 표시
        display_plotly_centered(fig_corr)
        
        # 상관관계 해석 추가
        corr_interpretation = ""
        if any(abs(val) > 0.7 for val in top_corrs.values):
            corr_interpretation += "💡 **강한 상관관계(|r| > 0.7)가 발견되었습니다.** 이는 해당 변수들이 서로 밀접하게 연관되어 있음을 의미합니다.\n\n"
        if any(0.3 < abs(val) < 0.7 for val in top_corrs.values):
            corr_interpretation += "📊 **중간 정도의 상관관계(0.3 < |r| < 0.7)가 있는 변수들이 있습니다.** 이들은 부분적으로 연관되어 있습니다.\n\n"
        if all(abs(val) < 0.3 for val in top_corrs.values):
            corr_interpretation += "⚠️ **모든 변수와 약한 상관관계(|r| < 0.3)를 보입니다.** 이는 선택한 변수가 다른 변수들과 뚜렷한 선형 관계가 없음을 의미할 수 있습니다.\n\n"
        
        st.markdown(corr_interpretation)
        
        # 상관관계 테이블
        st.subheader("상관계수 테이블")
        corr_table = pd.DataFrame({
            '변수': corr_data.index,
            '상관계수': corr_data.values
        }).sort_values('상관계수', key=abs, ascending=False).head(10)
        
        # 상관관계 강도 표시 추가
        corr_table['강도'] = corr_table['상관계수'].apply(
            lambda x: '강함 💪' if abs(x) > 0.7 else 
                     '중간 👌' if abs(x) > 0.3 else 
                     '약함 👎')
        
        st.table(corr_table.style.format({'상관계수': '{:.3f}'}))
        
        # 산점도 섹션 - Plotly로 변경
        st.subheader(f"{target_var}와(과) 주요 변수들의 산점도")
        
        # 상관관계가 높은 상위 변수들에 대해 산점도 그리기 (10개로 증가)
        top_vars = corr_data.abs().sort_values(ascending=False).head(10).index.tolist()
        
        # Plotly 서브플롯 생성 (3x4 그리드로 변경, 마지막 두 칸은 빈칸)
        fig_scatter = make_subplots(
            rows=3, cols=4,
            subplot_titles=[f'{var} vs {target_var} (r={data[[var, target_var]].corr().iloc[0, 1]:.2f})' for var in top_vars] + ['', ''],
            vertical_spacing=0.12,
            horizontal_spacing=0.05
        )
        
        # 각 변수에 대한 산점도 추가
        for i, var in enumerate(top_vars):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            # 상관계수 계산
            corr_val = data[[var, target_var]].corr().iloc[0, 1]
            
            # 산점도 추가
            fig_scatter.add_trace(
                go.Scatter(
                    x=data[var],
                    y=data[target_var],
                    mode='markers',
                    name=var,
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color='blue'
                    ),
                    hovertemplate=f'{var}: %{{x}}<br>{target_var}: %{{y}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            try:
                # 회귀선 추가를 위한 데이터 준비
                x_data = data[var].values
                y_data = data[target_var].values
                
                # NaN 값 제거
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[mask]
                y_data = y_data[mask]
                
                if len(x_data) > 1:  # 최소 2개 이상의 데이터 포인트가 필요
                    # 회귀선 계산
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    x_range = np.linspace(np.min(x_data), np.max(x_data), 100)
                    y_range = intercept + slope * x_range
                    
                    # 회귀선 추가
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode='lines',
                            name=f'회귀선 (r={corr_val:.2f})',
                            line=dict(color='red', width=2),
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
            except Exception as e:
                st.warning(f"'{var}' 변수의 회귀선 계산 중 오류 발생: {str(e)}")
        
        # 레이아웃 설정
        fig_scatter.update_layout(
            height=1000,  # 높이 증가
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # X축과 Y축 타이틀 설정
        for i, var in enumerate(top_vars):
            row = (i // 4) + 1
            col = (i % 4) + 1
            fig_scatter.update_xaxes(title_text=var, row=row, col=col)
            if col == 1:  # 왼쪽 열에만 Y축 레이블 표시
                fig_scatter.update_yaxes(title_text=target_var, row=row, col=col)
        
        # 그리드 추가
        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # 중앙에 표시 (너비 증가)
        display_plotly_centered(fig_scatter, width_pct=90)
        
        # 산점도 해석 추가
        with st.expander("📈 산점도 해석 방법"):
            st.markdown("""
            ### 산점도 해석 가이드
            
            1. **점들의 패턴 확인**:
               - **직선 형태로 모여 있음**: 선형 관계 존재
               - **곡선 형태**: 비선형 관계 존재
               - **무작위하게 퍼져 있음**: 관계 없음
            
            2. **회귀선(빨간색 선) 기울기**:
               - **양의 기울기**: 양의 상관관계
               - **음의 기울기**: 음의 상관관계
               - **수평에 가까움**: 관계 약함
            
            3. **점들의 밀집도**:
               - **회귀선 주변에 밀집**: 강한 상관관계
               - **넓게 퍼져 있음**: 약한 상관관계
            
            4. **이상치 확인**:
               - 대부분의 점들과 멀리 떨어진 점들은 이상치일 수 있으며, 관계 해석에 영향을 줄 수 있습니다.
            """)
        
        # 박스플롯 섹션 - Plotly로 변경
        st.subheader("박스플롯 분석")
        
        # 다중 선택 변수
        selected_vars = st.multiselect("변수 선택:", numeric_cols, default=[target_var])
        
        if selected_vars:
            st.write("각 변수별 박스플롯:")
            
            # 한 행에 최대 3개의 박스플롯을 표시
            cols_per_row = 3
            rows_needed = (len(selected_vars) + cols_per_row - 1) // cols_per_row
            
            # 행 단위로 반복
            for row_idx in range(rows_needed):
                # 각 행에 필요한 열 생성
                cols = st.columns(cols_per_row)
                
                # 현재 행에 표시할 변수들
                start_idx = row_idx * cols_per_row
                end_idx = min(start_idx + cols_per_row, len(selected_vars))
                row_vars = selected_vars[start_idx:end_idx]
                
                # 각 열에 변수 하나씩 표시
                for col_idx, var in enumerate(row_vars):
                    with cols[col_idx]:
                        # 개별 박스플롯 생성
                        fig_box = go.Figure()
                        
                        # 박스플롯 추가
                        fig_box.add_trace(
                            go.Box(
                                y=data[var],
                                name=var,
                                boxmean=True,  # 평균선 추가
                                boxpoints='outliers',  # 이상치만 점으로 표시
                                marker_color='lightseagreen',
                                line_color='darkblue',
                                hovertemplate='값: %{y}<extra></extra>'
                            )
                        )
                        
                        # 레이아웃 설정
                        fig_box.update_layout(
                            title=var,
                            height=400,
                            width=350,
                            margin=dict(l=10, r=10, t=40, b=20),
                            showlegend=False
                        )
                        
                        # 그리드 추가
                        fig_box.update_yaxes(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='LightGrey',
                            title='값'
                        )
                        
                        # 그래프 표시
                        st.plotly_chart(fig_box, use_container_width=True)
            
            # 박스플롯 해석 추가
            with st.expander("📦 박스플롯 해석 방법"):
                st.markdown("""
                ### 박스플롯 구성 요소
                """)
                
                st.image("./image/BOX_PLOT.png", caption="박스플롯의 주요 구성 요소")
                
                st.markdown("""
                1. **박스(Box)**: 1사분위수(Q1, 25%)에서 3사분위수(Q3, 75%)까지의 범위를 나타냅니다.
                   - 이 범위는 데이터의 중간 50%를 포함합니다.
                
                2. **중앙선(Median Line)**: 중앙값(50%)을 나타냅니다.
                   - 중앙값이 박스 내에서 중앙에 있으면 데이터가 대칭적입니다.
                   - 한쪽으로 치우쳐 있으면 데이터가 비대칭적(skewed)입니다.
                
                3. **수염(Whiskers)**: 박스 바깥쪽으로 뻗은 선으로, 정상 범위의 최소/최대값을 나타냅니다.
                   - 일반적으로 Q1 - 1.5*IQR 또는 최소값까지, Q3 + 1.5*IQR 또는 최대값까지 뻗습니다.
                
                4. **이상치(Outliers)**: 수염 바깥에 있는 점들로, 일반적인 분포에서 벗어난 값들입니다.
                   - 이상치가 많다면 데이터에 특이한 패턴이 있거나 문제가 있을 수 있습니다.
                """)
            
            # 기술통계량 표시
            st.subheader("기술통계량")
            stats_df = data[selected_vars].describe().T
            
            # 보기 좋게 형식 지정
            formatted_stats = stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                '25%': '{:.2f}',
                '50%': '{:.2f}',
                '75%': '{:.2f}',
                'max': '{:.2f}'
            })
            
            st.table(formatted_stats)

    # -----------------------------------------
    # 새로운 섹션: 범주형 데이터 빈도 분석 추가
    # -----------------------------------------
    st.divider() # 섹션 구분을 위한 선 추가
    st.subheader("📊 범주형 데이터 빈도 분석")

    # 범주형 데이터 분석 설명 추가
    with st.expander("🤔 범주형 데이터 빈도 분석이란?"):
        st.markdown("""
        범주형 데이터는 **문자열(텍스트)**이나 **정해진 카테고리**(예: '합격'/'불합격', 'A등급'/'B등급', '라인1'/'라인2')로 이루어진 데이터를 말합니다.
        
        **빈도 분석**은 선택한 컬럼(변수)에서 각각의 값(범주)들이 **몇 번씩 나타나는지(빈도수)**, 그리고 **전체 데이터에서 차지하는 비율**은 얼마인지 확인하는 분석입니다.
        
        **왜 필요한가요?**
        - 데이터의 **구성 비율**을 파악할 수 있습니다. (예: 전체 제품 중 불량품 비율)
        - 특정 항목의 **분포**를 확인할 수 있습니다. (예: 각 생산 라인별 생산량 분포)
        - 데이터 처리나 모델링 전에 **데이터의 기본적인 특성**을 이해하는 데 도움이 됩니다.
        
        """)

    # 데이터에서 범주형 또는 문자열 타입 컬럼만 선택
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        st.warning("데이터에 분석할 범주형 또는 문자열 컬럼이 없습니다.")
    else:
        # 범주형 변수 선택
        selected_cat_var = st.selectbox(
            "분석할 범주형 변수 선택:",
            options=categorical_cols,
            index=0 # 기본값으로 첫 번째 범주형 변수 선택
        )
        
        if selected_cat_var:
            st.write(f"### 🔹 '{selected_cat_var}' 변수 분석 결과")
            
            # 선택된 변수의 빈도수 및 비율 계산
            value_counts = data[selected_cat_var].value_counts()
            value_percentages = data[selected_cat_var].value_counts(normalize=True) * 100
            
            # 결과를 보기 좋게 DataFrame으로 만듦
            freq_df = pd.DataFrame({
                '빈도수': value_counts,
                '비율 (%)': value_percentages
            })
            freq_df.index.name = '범주' # 인덱스 이름 설정
            
            # 결과 테이블 표시
            st.dataframe(freq_df.style.format({'비율 (%)': '{:.2f}%'}))
            
            # Plotly 막대 그래프 생성
            fig_bar = go.Figure()
            
            fig_bar.add_trace(
                go.Bar(
                    x=freq_df.index,
                    y=freq_df['빈도수'],
                    text=freq_df['빈도수'], # 막대 위에 빈도수 표시
                    textposition='auto', # 텍스트 위치 자동 조정
                    marker_color=px.colors.qualitative.Pastel, # 색상 설정
                    hovertemplate='범주: %{x}<br>빈도수: %{y}<extra></extra>'
                )
            )
            
            # 그래프 레이아웃 설정
            fig_bar.update_layout(
                title=f"'{selected_cat_var}'의 빈도수 분포", # f-string 시작 따옴표 수정
                xaxis_title='범주',
                yaxis_title='빈도수',
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_tickangle=-45 # x축 레이블 기울임 (겹침 방지)
            )
            
            # 그리드 추가
            fig_bar.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            
            # 중앙에 표시
            display_plotly_centered(fig_bar, width_pct=70)

    # -----------------------------------------
    # 새로운 섹션: 집단 간 비교 분석 추가
    # -----------------------------------------
    st.divider()
    st.subheader("📈 집단 간 비교 분석")

    # --- 샘플 데이터 안내 메시지 추가 ---
    st.info("""
    **💡 분석 기능을 테스트해보고 싶으신가요?**
    
    데이터 업로드 페이지(📊1_데이터_업로드.py)에서 T-검정과 ANOVA 샘플 데이터를 다운로드할 수 있습니다.
    
    - **T-검정 샘플 데이터**: '라인'(A, B)과 '수율' 데이터가 포함된 CSV 파일입니다. 두 그룹 비교(T-검정)에 사용됩니다.
    - **ANOVA 샘플 데이터**: '공급업체'(X, Y, Z)와 '강도' 데이터가 포함된 CSV 파일입니다. 세 그룹 이상 비교(ANOVA)에 사용됩니다.
    """)
    st.markdown("--- ") # 구분선 추가
    # --- 샘플 데이터 안내 메시지 끝 ---

    analysis_type = st.radio(
        "어떤 비교를 하고 싶으신가요?",
        [
            "1️⃣ 특정 항목(숫자)을 **두 그룹** 간에 비교하기 (예: 라인 A vs 라인 B의 수율 비교)",
            "2️⃣ 특정 항목(숫자)을 **세 그룹 이상** 간에 비교하기 (예: 공급업체 A/B/C 간 원자재 강도 비교)",
        ],
        index=None, # 선택하지 않은 상태로 시작
        key="analysis_type_radio", # 고유 키 할당
        help="""
        - **두 그룹 비교**: 독립적인 두 집단의 평균 차이를 봅니다 (독립표본 T-검정).
        - **세 그룹 이상 비교**: 독립적인 여러 집단의 평균 차이를 봅니다 (분산분석 ANOVA).
        """
    )
    
    # 숫자형 변수 목록 미리 준비
    numeric_cols_compare = data.select_dtypes(include=np.number).columns.tolist()
    # 범주형 변수 목록 (이전 섹션에서 가져옴)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 1. 두 그룹 비교 (독립표본 T-검정) --- 
    if analysis_type and analysis_type.startswith("1️⃣"):
        st.write("### 1. 두 그룹 비교 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            numeric_var_ttest = st.selectbox("① 평균 비교 대상 변수 선택:", numeric_cols_compare, index=None, key="num_ttest")
        
        # 그룹 변수 선택 (고유값 2개인 범주형 변수만 필터링)
        two_groups_cols = [col for col in categorical_cols if data[col].nunique() == 2]
        
        with col2:
            group_var_ttest = st.selectbox("② 두 그룹 기준 변수 선택:", two_groups_cols, index=None, key="group_ttest", 
                                        help="정확히 두 개의 그룹(값)으로 나누는 변수를 선택하세요.")
            
        if numeric_var_ttest and group_var_ttest:
            st.write(f"### 🔹 '{numeric_var_ttest}'의 '{group_var_ttest}' 그룹 간 비교 결과")
            
            # 데이터 준비
            group_values = data[group_var_ttest].unique()
            group1_data = data[data[group_var_ttest] == group_values[0]][numeric_var_ttest].dropna()
            group2_data = data[data[group_var_ttest] == group_values[1]][numeric_var_ttest].dropna()
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                st.warning("각 그룹에 최소 2개 이상의 데이터가 있어야 T-검정을 수행할 수 있습니다. (각 그룹별 데이터 개수를 확인해주세요)")
            else:
                # 정규성 검증 (Shapiro-Wilk 검정)
                st.write("#### 1. 정규성 검증")
                _, p_value1 = stats.shapiro(group1_data)
                _, p_value2 = stats.shapiro(group2_data)
                
                # 정규성 검증 결과 표시
                st.write(f"**그룹 '{group_values[0]}' 정규성 검증:** p-value = {p_value1:.4f}")
                st.write(f"**그룹 '{group_values[1]}' 정규성 검증:** p-value = {p_value2:.4f}")
                
                # 정규성 판단 (p-value > 0.05이면 정규성 만족)
                is_normal = p_value1 > 0.05 and p_value2 > 0.05
                
                if is_normal:
                    st.success("✅ **정규성 검증 결과:** 두 그룹 모두 정규성을 만족합니다. (p > 0.05)")
                    st.write("#### 2. 독립표본 T-검정 수행")
                    
                    # 독립표본 T-검정 수행
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's T-test (등분산 가정 안함)
                    
                    # 결과 해석
                    st.write(f"**T-검정 결과:** T-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")
                    st.markdown("**결과 해석:**")
                    if p_value < 0.05:
                        st.success(f"✅ **결론:** 두 그룹('{group_values[0]}', '{group_values[1]}') 간 '{numeric_var_ttest}' 평균에는 **통계적으로 의미있는 차이가 있습니다** (p < 0.05). 그룹별 평균값을 확인해보세요!")
                    else:
                        st.info(f"ℹ️ **결론:** 두 그룹('{group_values[0]}', '{group_values[1]}') 간 '{numeric_var_ttest}' 평균 차이가 **통계적으로 의미있다고 보기는 어렵습니다** (p ≥ 0.05). 우연에 의한 차이일 수 있습니다.")
                    st.caption("👉 p-value가 0.05보다 작으면 '차이가 있다'고 판단하는 것이 일반적입니다.")
                else:
                    st.warning("⚠️ **정규성 검증 결과:** 하나 이상의 그룹이 정규성을 만족하지 않습니다. (p ≤ 0.05)")
                    st.write("#### 2. Mann-Whitney U 검정 수행 (비모수 검정)")
                    
                    # Mann-Whitney U 검정 수행
                    _, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    
                    # 결과 해석
                    st.write(f"**Mann-Whitney U 검정 결과:** p-value = {p_value:.4f}")
                    st.markdown("**결과 해석:**")
                    if p_value < 0.05:
                        st.success(f"✅ **결론:** 두 그룹('{group_values[0]}', '{group_values[1]}') 간 '{numeric_var_ttest}' 분포에는 **통계적으로 의미있는 차이가 있습니다** (p < 0.05). 그룹별 중앙값을 확인해보세요!")
                    else:
                        st.info(f"ℹ️ **결론:** 두 그룹('{group_values[0]}', '{group_values[1]}') 간 '{numeric_var_ttest}' 분포 차이가 **통계적으로 의미있다고 보기는 어렵습니다** (p ≥ 0.05). 그룹 간 차이는 우연일 수 있습니다.")
                    st.caption("👉 p-value가 0.05보다 작으면 '차이가 있다'고 판단하는 것이 일반적입니다.")
                
                # 그룹별 기술 통계량
                st.write("**그룹별 기술 통계량:**")
                stats_summary = data.groupby(group_var_ttest)[numeric_var_ttest].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                st.dataframe(stats_summary.style.format({'mean': '{:.2f}', 'std': '{:.2f}', 'min': '{:.2f}', 'max': '{:.2f}'}))
                
                # 그룹별 박스 플롯 (Plotly)
                fig_box_ttest = px.box(data, x=group_var_ttest, y=numeric_var_ttest, 
                                        color=group_var_ttest, # 그룹별 색상 구분
                                        title=f"'{numeric_var_ttest}'의 그룹별 분포",
                                        labels={numeric_var_ttest: f"{numeric_var_ttest} 값", group_var_ttest: "그룹"},
                                        points="all") # 모든 점 표시
                fig_box_ttest.update_layout(height=500)
                display_plotly_centered(fig_box_ttest)

    # --- 2. 세 그룹 이상 비교 (ANOVA) --- 
    elif analysis_type and analysis_type.startswith("2️⃣"):
        st.write("### 2. 세 그룹 이상 비교 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            numeric_var_anova = st.selectbox("① 평균 비교 대상 변수 선택:", numeric_cols_compare, index=None, key="num_anova")
            
        # 그룹 변수 선택 (고유값 3개 이상인 범주형 변수만 필터링)
        multi_groups_cols = [col for col in categorical_cols if data[col].nunique() >= 3]
        
        with col2:
            group_var_anova = st.selectbox("② 세 그룹 이상 기준 변수 선택:", multi_groups_cols, index=None, key="group_anova",
                                         help="세 개 이상의 그룹(값)으로 나누는 변수를 선택하세요.")
            
        if numeric_var_anova and group_var_anova:
            st.write(f"### 🔹 '{numeric_var_anova}'의 '{group_var_anova}' 그룹 간 비교 결과 (ANOVA)")
            
            # 데이터 준비
            groups = data[group_var_anova].unique()
            group_data_list = [data[data[group_var_anova] == group][numeric_var_anova].dropna() for group in groups]
            
            # 각 그룹별 데이터 개수 확인
            if any(len(group_data) < 2 for group_data in group_data_list):
                 st.warning("각 그룹에 최소 2개 이상의 데이터가 있어야 ANOVA 검정을 수행할 수 있습니다. (각 그룹별 데이터 개수를 확인해주세요)")
            else:
                # 정규성 검증 (Shapiro-Wilk 검정)
                st.write("#### 1. 정규성 검증")
                
                # 각 그룹별 정규성 검증 결과 저장
                normality_results = []
                for i, group_data in enumerate(group_data_list):
                    _, p_value = stats.shapiro(group_data)
                    normality_results.append((groups[i], p_value))
                    st.write(f"**그룹 '{groups[i]}' 정규성 검증:** p-value = {p_value:.4f}")
                
                # 정규성 판단 (모든 그룹의 p-value > 0.05이면 정규성 만족)
                is_normal = all(p_value > 0.05 for _, p_value in normality_results)
                
                if is_normal:
                    st.success("✅ **정규성 검증 결과:** 모든 그룹이 정규성을 만족합니다. (p > 0.05)")
                    st.write("#### 2. ANOVA 검정 수행")
                    
                    # ANOVA 검정 수행
                    f_stat, p_value = stats.f_oneway(*group_data_list)
                    
                    # 결과 해석
                    st.write(f"**ANOVA 검정 결과:** F-statistic = {f_stat:.3f}, p-value = {p_value:.4f}")
                    st.markdown("**결과 해석:**")
                    if p_value < 0.05:
                        st.success(f"✅ **결론:** '{group_var_anova}' 그룹들 간 '{numeric_var_anova}' 평균에는 **적어도 하나 이상의 의미있는 차이가 존재합니다** (p < 0.05).")
                    else:
                        st.info(f"ℹ️ **결론:** '{group_var_anova}' 그룹들 간 '{numeric_var_anova}' 평균 차이가 **통계적으로 의미있다고 보기는 어렵습니다** (p ≥ 0.05). 그룹 간 차이는 우연일 수 있습니다.")
                    st.caption("👉 p-value가 0.05보다 작으면 '그룹 간 차이가 있다'고 판단하는 것이 일반적입니다.")
                else:
                    st.warning("⚠️ **정규성 검증 결과:** 하나 이상의 그룹이 정규성을 만족하지 않습니다. (p ≤ 0.05)")
                    st.write("#### 2. Kruskal-Wallis H 검정 수행 (비모수 검정)")
                    
                    # Kruskal-Wallis H 검정 수행
                    h_stat, p_value = stats.kruskal(*group_data_list)
                    
                    # 결과 해석
                    st.write(f"**Kruskal-Wallis H 검정 결과:** H-statistic = {h_stat:.3f}, p-value = {p_value:.4f}")
                    st.markdown("**결과 해석:**")
                    if p_value < 0.05:
                        st.success(f"✅ **결론:** '{group_var_anova}' 그룹들 간 '{numeric_var_anova}' 분포에는 **적어도 하나 이상의 의미있는 차이가 존재합니다** (p < 0.05).")
                    else:
                        st.info(f"ℹ️ **결론:** '{group_var_anova}' 그룹들 간 '{numeric_var_anova}' 분포 차이가 **통계적으로 의미있다고 보기는 어렵습니다** (p ≥ 0.05). 그룹 간 차이는 우연일 수 있습니다.")
                    st.caption("👉 p-value가 0.05보다 작으면 '그룹 간 차이가 있다'고 판단하는 것이 일반적입니다.")

                # 그룹별 기술 통계량
                st.write("**그룹별 기술 통계량:**")
                stats_summary_anova = data.groupby(group_var_anova)[numeric_var_anova].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                st.dataframe(stats_summary_anova.style.format({'mean': '{:.2f}', 'std': '{:.2f}', 'min': '{:.2f}', 'max': '{:.2f}'}))
                
                # 그룹별 박스 플롯 (Plotly)
                fig_box_anova = px.box(data, x=group_var_anova, y=numeric_var_anova, 
                                       color=group_var_anova, # 그룹별 색상 구분
                                       title=f"'{numeric_var_anova}'의 그룹별 분포",
                                       labels={numeric_var_anova: f"{numeric_var_anova} 값", group_var_anova: "그룹"},
                                       points="all") # 모든 점 표시
                fig_box_anova.update_layout(height=500)
                display_plotly_centered(fig_box_anova)

                # 정규성을 만족하는 경우의 사후분석
                if is_normal and p_value < 0.05:
                    # Tukey's HSD 사후분석 수행
                    st.write("#### 3. Tukey's HSD 사후분석")
                    st.write("각 그룹 쌍별로 평균 차이를 비교합니다.")
                    
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    
                    # 데이터 준비
                    values = data[numeric_var_anova].values
                    groups = data[group_var_anova].values
                    
                    # Tukey's HSD 수행
                    tukey = pairwise_tukeyhsd(values, groups)
                    
                    # 결과를 데이터프레임으로 변환
                    tukey_df = pd.DataFrame(
                        data=tukey._results_table.data[1:],
                        columns=['그룹1', '그룹2', '평균차이', '하한', '상한', 'p-value', '유의성']
                    )
                    
                    # p-value 형식 지정 및 유의성 표시 수정
                    tukey_df['p-value'] = tukey_df['p-value'].astype(float)
                    tukey_df['유의성'] = tukey_df['p-value'].apply(lambda x: '유의미한 차이 있음 ✓' if x < 0.05 else '차이 없음')
                    
                    # 결과 표시
                    st.write("**그룹 간 차이 분석 결과:**")
                    st.dataframe(
                        tukey_df.style.format({
                            '평균차이': '{:.3f}',
                            '하한': '{:.3f}',
                            '상한': '{:.3f}',
                            'p-value': '{:.4f}'
                        })
                    )
                    
                    # 해석 가이드 추가
                    st.info("""
                    **🔍 해석 방법:**
                    - **p-value < 0.05**: 해당 두 그룹 간에 통계적으로 유의미한 차이가 있음
                    - **평균차이**: 그룹1과 그룹2의 평균 차이 (그룹1 - 그룹2)
                    - **하한/상한**: 평균 차이의 95% 신뢰구간
                    """)
                
                # 정규성을 만족하지 않는 경우의 사후분석
                elif not is_normal and p_value < 0.05:
                    # Mann-Whitney U 검정으로 쌍별 비교 수행
                    st.write("#### 3. Mann-Whitney U 사후분석")
                    st.write("각 그룹 쌍별로 분포 차이를 비교합니다.")
                    
                    # 모든 가능한 그룹 쌍 생성
                    from itertools import combinations
                    group_pairs = list(combinations(groups, 2))
                    
                    # 결과를 저장할 리스트
                    pair_results = []
                    
                    # 각 쌍에 대해 Mann-Whitney U 검정 수행
                    for group1, group2 in group_pairs:
                        data1 = data[data[group_var_anova] == group1][numeric_var_anova]
                        data2 = data[data[group_var_anova] == group2][numeric_var_anova]
                        
                        # Mann-Whitney U 검정 수행
                        stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # 결과 저장
                        pair_results.append({
                            '그룹1': group1,
                            '그룹2': group2,
                            '중앙값1': data1.median(),
                            '중앙값2': data2.median(),
                            '중앙값 차이': data1.median() - data2.median(),
                            'p-value': p_val,
                            '유의성': '유의미한 차이 있음 ✓' if p_val < 0.05 else '차이 없음'
                        })
                    
                    # 결과를 데이터프레임으로 변환
                    pair_results_df = pd.DataFrame(pair_results)
                    
                    # 결과 표시
                    st.write("**그룹 간 차이 분석 결과:**")
                    st.dataframe(
                        pair_results_df.style.format({
                            '중앙값1': '{:.3f}',
                            '중앙값2': '{:.3f}',
                            '중앙값 차이': '{:.3f}',
                            'p-value': '{:.4f}'
                        })
                    )
                    
                    # 해석 가이드 추가
                    st.info("""
                    **🔍 해석 방법:**
                    - **p-value < 0.05**: 해당 두 그룹 간에 통계적으로 유의미한 차이가 있음
                    - **중앙값 차이**: 그룹1과 그룹2의 중앙값 차이 (그룹1 - 그룹2)
                    - 비모수 검정이므로 평균 대신 중앙값을 사용하여 비교
                    """)

else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")

# 페이지 하단 소개
st.markdown("---")
st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ")