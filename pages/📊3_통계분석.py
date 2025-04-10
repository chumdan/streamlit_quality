import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        
        # 상관관계가 높은 상위 변수들에 대해 산점도 그리기
        top_vars = corr_data.abs().sort_values(ascending=False).head(4).index.tolist()
        
        # Plotly 서브플롯 생성 (2x2 그리드)
        fig_scatter = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{var} vs {target_var} (r={data[[var, target_var]].corr().iloc[0, 1]:.2f})' for var in top_vars]
        )
        
        # 각 변수에 대한 산점도 추가
        for i, var in enumerate(top_vars):
            row = i // 2 + 1
            col = i % 2 + 1
            
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
            
            # 회귀선 추가를 위한 데이터 준비
            x_range = np.linspace(data[var].min(), data[var].max(), 100)
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[var], data[target_var])
            y_range = intercept + slope * x_range
            
            # 회귀선 추가
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'회귀선 (r={corr_val:.2f})',
                    line=dict(color='red'),
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
        
        # 레이아웃 설정
        fig_scatter.update_layout(
            height=700,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # X축과 Y축 타이틀 설정
        for i, var in enumerate(top_vars):
            row = i // 2 + 1
            col = i % 2 + 1
            fig_scatter.update_xaxes(title_text=var, row=row, col=col)
            if col == 1:  # 왼쪽 열에만 Y축 레이블 표시
                fig_scatter.update_yaxes(title_text=target_var, row=row, col=col)
        
        # 그리드 추가
        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # 중앙에 표시
        display_plotly_centered(fig_scatter, width_pct=70)
        
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
                
                ![박스플롯 설명](https://miro.medium.com/max/1400/1*2c21SkzJMf3frPXPAR_gZA.png)
                
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
else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")