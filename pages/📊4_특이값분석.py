import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 테마 색상 정의
PRIMARY_COLOR = "#004F9F"  # 중근당 BLUE
GOLD_COLOR = "#C0A548"  
SILVER_COLOR = "#B5B5B5"
BLACK_COLOR = "#000000"

st.title("3. 특이값 분석")

# 특이값 분석 개념 설명 추가
with st.expander("📚 특이값(이상치) 분석이란?"):
    st.markdown("""
    ### 특이값/이상치(Outlier)란?
    특이값은 전체 데이터 패턴에서 벗어나 비정상적으로 보이는 데이터 포인트입니다. 이는 데이터 수집 오류, 측정 오류, 자연적 변동 등으로 인해 발생할 수 있습니다.
    
    ### Z-점수(Z-score) 방법
    - 각 데이터 포인트가 평균으로부터 몇 표준편차만큼 떨어져 있는지 계산합니다.
    - Z-점수 = (측정값 - 평균) / 표준편차
    - **Z-점수의 의미**: 0에 가까울수록 평균에 가깝고, ±1, ±2, ±3은 각각 해당 표준편차만큼 평균에서 벗어났음을 의미합니다.
    
    ### Z-점수 해석 가이드
    
    | Z-점수 범위 | 해석 | 특이값 여부 |
    |------------|------|------------|
    | -1 ~ +1    | 평균 주변 (전체 데이터의 약 68%) | 정상 |
    | -2 ~ -1 또는 +1 ~ +2 | 평균에서 약간 벗어남 (전체 데이터의 약 27%) | 정상 |
    | -3 ~ -2 또는 +2 ~ +3 | 평균에서 상당히 벗어남 (전체 데이터의 약 4.5%) | 주의 필요 |
    | < -3 또는 > +3 | 평균에서 크게 벗어남 (전체 데이터의 약 0.3%) | 특이값 |
    
    ### 배치 기반 특이값 분석의 활용
    
    - **품질 문제 추적**: 불량 제품이 발생한 배치에서 어떤 공정 변수가 특이했는지 파악
    - **공정 개선**: 자주 특이값을 보이는 변수를 중심으로 공정 안정화 방안 모색
    - **근본 원인 분석**: 여러 변수가 동시에 특이값을 보일 경우 공통 원인 파악
    """)

# 데이터 확인
if 'data' in st.session_state and st.session_state.data is not None:
    data = st.session_state.data
    
    # 배치 설정 부분 수정
    st.subheader("배치 설정")

    # 모든 컬럼 목록 가져오기
    all_columns = data.columns.tolist()

    # 배치 ID 컬럼이 있는지 확인
    potential_batch_cols = [col for col in all_columns if 'batch' in col.lower() or 'id' in col.lower() or 'lot' in col.lower() or '번호' in col]

    # 배치 식별 방법 선택 (라디오 버튼으로 변경)
    batch_id_method = st.radio(
        "배치 식별 방법:",
        ["데이터 인덱스를 배치 ID로 사용", "특정 컬럼을 배치 ID로 사용"],
        index=0 if len(potential_batch_cols) == 0 else 1,
        help="배치를 어떻게 식별할지 선택하세요."
    )

    use_index_as_batch = (batch_id_method == "데이터 인덱스를 배치 ID로 사용")

    # 배치 ID로 특정 컬럼을 사용할 경우에만 컬럼 선택 표시
    batch_col = None
    if not use_index_as_batch:
        batch_col = st.selectbox(
            "배치 ID 컬럼 선택:",
            options=all_columns,
            index=all_columns.index(potential_batch_cols[0]) if len(potential_batch_cols) > 0 else 0,
            help="각 행을 식별하는 고유 ID 컬럼을 선택하세요 (예: 배치번호, LOT_ID 등)"
        )

    # 배치 이름에 추가 정보 사용 여부
    use_name_column = st.checkbox("배치 이름 표시에 사용할 추가 컬럼 선택", 
                                  help="배치 ID 외에 제품명, 생산일자 등 배치를 더 쉽게 식별할 수 있는 컬럼을 선택합니다.")

    # 추가 정보 컬럼 선택
    name_column = None
    if use_name_column:
        name_column = st.selectbox(
            "배치 이름 컬럼 선택:",
            options=all_columns,
            index=0,
            help="이 컬럼의 값이 배치 ID와 함께 표시됩니다."
        )
    
    if not use_index_as_batch:
        data_analysis = data.copy()
        
        # 배치 표시 이름 생성 (ID + 이름)
        if name_column and name_column in data.columns:
            data_analysis['배치_표시명'] = data_analysis[batch_col].astype(str) + " (" + data_analysis[name_column].astype(str) + ")"
            batch_display_dict = dict(zip(data_analysis[batch_col], data_analysis['배치_표시명']))
        else:
            batch_display_dict = dict(zip(data_analysis[batch_col], data_analysis[batch_col]))
        
        data_analysis.set_index(batch_col, inplace=True)
        batch_ids = data_analysis.index.tolist()
        
        st.info(f"'{batch_col}' 컬럼을 배치 ID로 사용합니다." + 
                (f" '{name_column}' 컬럼을 배치 이름으로 함께 표시합니다." if name_column else ""))
    else:
        st.info("현재 데이터 인덱스를 배치 ID로 사용합니다.")
        data_analysis = data.copy()
        batch_ids = data_analysis.index.tolist()
        
        # 배치 표시 이름 생성 (인덱스 + 이름)
        if name_column and name_column in data.columns:
            data_analysis['배치_표시명'] = data_analysis.index.astype(str) + " (" + data_analysis[name_column].astype(str) + ")"
            batch_display_dict = dict(zip(data_analysis.index, data_analysis['배치_표시명']))
        else:
            batch_display_dict = dict(zip(batch_ids, batch_ids))
    
    # Z-점수 임계값 설정
    st.subheader("특이값 분석 설정")
    
    # Z-점수 임계값 설정에 도움말 강화
    z_threshold = st.slider(
        "Z-점수 임계값 설정 (±):",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="이 값보다 큰 절대 Z-점수를 가진 값을 특이값으로 간주합니다."
    )
    
    # 변수 선택 (숫자형 데이터만)
    numeric_cols = data_analysis.select_dtypes(include=[np.number]).columns.tolist()
    
    # 변수 선택
    selected_vars = st.multiselect(
        "분석할 변수 선택 (기본: 모든 숫자형 변수):",
        options=numeric_cols,
        default=numeric_cols[:min(8, len(numeric_cols))]  # 기본적으로 최대 8개 변수 선택
    )
    
    if not selected_vars:
        st.warning("최소한 하나 이상의 변수를 선택하세요.")
        selected_vars = numeric_cols[:1] if numeric_cols else []
    
    if selected_vars:
        # 모든 변수에 대한 Z-score 계산
        z_scores = pd.DataFrame(index=data_analysis.index)
        
        for col in selected_vars:
            z_scores[f"{col}_zscore"] = stats.zscore(data_analysis[col], nan_policy='omit')
        
        # 각 배치별로 최대 Z-score 절대값 계산
        max_abs_zscores = pd.Series(index=data_analysis.index, dtype=float)
        
        for batch in z_scores.index:
            batch_zscores = []
            for var in selected_vars:
                z_col = f"{var}_zscore"
                if z_col in z_scores.columns:
                    batch_zscores.append(abs(z_scores.loc[batch, z_col]))
            
            if batch_zscores:
                max_abs_zscores[batch] = max(batch_zscores)
            else:
                max_abs_zscores[batch] = 0
        
        # 특이 배치 식별 (Z-score가 임계값을 초과하는 배치)
        unusual_batches = max_abs_zscores[max_abs_zscores > z_threshold].sort_values(ascending=False)
        
        # 배치 선택을 위한 옵션
        st.subheader("관심 배치 선택")
        
        # 배치 선택 방법
        batch_selection_method = st.radio(
            "배치 선택 방법:",
            ["특이값이 있는 배치만 보기", "모든 배치에서 선택"]
        )
        
        if batch_selection_method == "특이값이 있는 배치만 보기":
            if len(unusual_batches) > 0:
                selected_batch = st.selectbox(
                    "분석할 배치 선택 (Z-score 내림차순):",
                    options=unusual_batches.index.tolist(),
                    format_func=lambda x: f"{x} (최대 Z-score: {unusual_batches[x]:.2f})"
                )
                
                st.success(f"배치 '{selected_batch}'의 최대 Z-score는 {unusual_batches[selected_batch]:.2f}입니다.")
            else:
                st.warning(f"Z-score 임계값 {z_threshold}를 초과하는 배치가, 없습니다.")
                selected_batch = st.selectbox(
                    "분석할 배치 선택:",
                    options=batch_ids
                )
        else:
            selected_batch = st.selectbox(
                "분석할 배치 선택:",
                options=batch_ids
            )
        
        # 선택된 배치 분석
        if selected_batch:
            st.header(f"배치 '{selected_batch}'의 변수별 Z-score 분석")
            
            # 선택된 배치의 Z-score 데이터 추출
            batch_zscores = {}
            for var in selected_vars:
                z_col = f"{var}_zscore"
                if z_col in z_scores.columns:
                    batch_zscores[var] = z_scores.loc[selected_batch, z_col]
            
            # 결과 요약
            zscore_df = pd.DataFrame({
                '변수': list(batch_zscores.keys()),
                'Z-score': list(batch_zscores.values())
            })
            zscore_df['특이값 여부'] = abs(zscore_df['Z-score']) > z_threshold
            zscore_df = zscore_df.sort_values(by='Z-score', key=abs, ascending=False)
            
            # 특이값 요약
            outlier_count = zscore_df['특이값 여부'].sum()
            if outlier_count > 0:
                st.warning(f"배치 '{selected_batch}'에서 {outlier_count}개 변수가 특이값으로 감지되었습니다.")
            else:
                st.success(f"배치 '{selected_batch}'에서 특이값이 없습니다.")
            
            # 변수 Z-score 테이블
            st.markdown("### 변수별 Z-score")
            st.dataframe(
                zscore_df.style.format({'Z-score': '{:.2f}'})
                         .applymap(lambda x: 'background-color: #ffcccc' if x else '', subset=['특이값 여부'])
                         .set_properties(**{'text-align': 'left'})
            )
            
            # 선택된 배치의 변수별 Z-score 시각화
            st.markdown(f"### 배치 '{selected_batch}'의 변수별 분포 및 Z-score")
            
            # 그래프 열 수 계산 - 한 행에 4개만 배치하여 더 넓게 표시
            max_cols_per_row = 4  # 6에서 4로 변경하여 더 넓게 표시
            n_cols = min(max_cols_per_row, len(selected_vars))
            n_rows = (len(selected_vars) + max_cols_per_row - 1) // max_cols_per_row  # 올림 나눗셈
            
            # 서브플롯 생성 - 간격 추가
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=[f"{var} (Z: {batch_zscores[var]:.2f})" for var in selected_vars],  # 제목 간소화
                horizontal_spacing=0.08,  # 수평 간격 넓게
                vertical_spacing=0.1      # 수직 간격 넓게
            )
            
            # 각 변수별 그래프 추가
            for i, var in enumerate(selected_vars):
                row = i // n_cols + 1  # 행 인덱스
                col = i % n_cols + 1   # 열 인덱스
                
                # 변수값 추출
                var_values = data_analysis[var].dropna()
                
                # 배치의 값
                batch_value = data_analysis.loc[selected_batch, var]
                
                # Z-score 계산
                z_score = batch_zscores[var]
                
                # 특이값 여부 확인
                is_outlier = abs(z_score) > z_threshold
                
                # 박스 플롯 색상 설정
                box_color = 'red' if is_outlier else 'black'
                box_fillcolor = 'rgba(255,0,0,0.1)' if is_outlier else 'rgba(255,255,255,0)'
                
                # 스트립 플롯 추가
                fig.add_trace(
                    go.Box(
                        x=var_values,
                        name=var,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(color='gray', size=5, opacity=0.6),
                        line=dict(color=box_color, width=2 if is_outlier else 1),
                        fillcolor=box_fillcolor
                    ),
                    row=row, col=col
                )
                
                # 선택된 배치 포인트 추가
                fig.add_trace(
                    go.Scatter(
                        x=[batch_value],
                        y=[0],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='circle'),
                        name=selected_batch,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Z-score 텍스트 추가
                z_score_text = f"Z-score: {z_score:.2f}"
                z_score_color = 'red' if is_outlier else 'black'
                z_score_bgcolor = 'rgba(255,0,0,0.1)' if is_outlier else 'white'
                
                fig.add_annotation(
                    x=batch_value,
                    y=0.15,
                    text=z_score_text,
                    showarrow=False,
                    font=dict(size=12, color=z_score_color),
                    bgcolor=z_score_bgcolor,
                    bordercolor=box_color,
                    borderwidth=1,
                    row=row, col=col
                )
                
                # 특이값 표시선 추가
                if is_outlier:
                    fig.add_shape(
                        type="line",
                        x0=var_values.min(),
                        y0=0,
                        x1=var_values.max(),
                        y1=0,
                        line=dict(color="red", width=1, dash="dash"),
                        row=row, col=col
                    )
            
            # 레이아웃 조정
            fig.update_layout(
                height=350 * n_rows,      # 높이 약간 증가
                width=1200,               # 너비 더 넓게 설정
                title=f"배치 '{selected_batch}'의 변수별 분포 분석 (빨간점: 현재 배치 값, 빨간색 테두리: 특이값)",
                showlegend=False,
                margin=dict(l=20, r=20, t=100, b=30)  # 여백 줄이기
            )
            
            # X축 제목 스타일 설정
            fig.update_xaxes(
                title_font=dict(size=10),  # 축 제목 폰트 크기 작게
                tickfont=dict(size=9)      # 눈금 폰트 크기 작게
            )
            
            # 서브플롯 제목 스타일 설정
            for i, var in enumerate(selected_vars):
                z_score = batch_zscores[var]
                is_outlier = abs(z_score) > z_threshold
                title_color = 'red' if is_outlier else 'black'
                
                fig['layout']['annotations'][i]['font'] = dict(size=12, color=title_color)
                fig['layout']['annotations'][i]['text'] = f"{var} (Z: {z_score:.2f})"
            
            # 그래프 출력
            st.plotly_chart(fig)
            
            # 종합 해석 및 조언
            st.markdown("### 분석 결과 해석")
            
            # 특이 변수 목록
            outlier_vars = zscore_df[zscore_df['특이값 여부'] == True]['변수'].tolist()
            
            if outlier_vars:
                # 특이 변수 정보
                st.warning(f"### 이 배치에서 발견된 특이 변수:")
                
                # 특이 변수 목록 표시
                outlier_text = ', '.join([f"{var} (Z-score: {batch_zscores[var]:.2f})" for var in outlier_vars])
                st.markdown(f"**{outlier_text}**")
                
                # 가능한 원인
                st.markdown("#### 가능한 원인:")
                st.markdown("""
                1. **공정 변동**: 위 변수들의 공정 파라미터가 정상 범위를 벗어났을 수 있습니다.
                2. **측정 오류**: 센서 또는 측정 장비 오작동으로 인한 측정값 이상일 수 있습니다.
                3. **외부 요인**: 환경 조건 변화(온도, 습도 등)가 영향을 미쳤을 수 있습니다.
                """)
                
                # 권장 조치
                st.markdown("#### 권장 조치:")
                st.markdown("""
                1. **특이 변수 검증**: 측정값이 실제로 정확한지 확인하세요.
                2. **공정 로그 확인**: 해당 배치 생산 시 공정 로그를 검토하여 이상 징후가 있었는지 확인하세요.
                3. **품질 테스트**: 이 배치의 최종 제품 품질에 문제가 있는지 확인하세요.
                """)
                
                # 가장 큰 영향을 미친 변수 분석
                worst_var = outlier_vars[0]
                st.info(f"### 가장 큰 특이값을 가진 변수: {worst_var}")
                st.markdown(f"**Z-score: {batch_zscores[worst_var]:.2f}**")
                
                # 특이값 설명
                st.markdown(f"이 변수는 평균보다 **{abs(batch_zscores[worst_var]):.1f}배** 표준편차만큼 {'높습니다' if batch_zscores[worst_var] > 0 else '낮습니다'}.")
                
                # 쉬운 설명
                st.markdown("#### 쉽게 설명하면:")
                st.markdown(f"""
                - 평균적인 배치에서 이 변수의 값은 {data_analysis[worst_var].mean():.2f} 정도입니다.
                - 이 배치에서는 {data_analysis.loc[selected_batch, worst_var]:.2f}로, {'정상보다 상당히 높습니다.' if batch_zscores[worst_var] > 0 else '정상보다 상당히 낮습니다.'}
                - 이런 상황은 전체 배치 중 약 {(1 - stats.norm.cdf(abs(batch_zscores[worst_var]))) * 2 * 100:.1f}% 정도만 발생합니다.
                """)
                
                # 가능한 조치
                st.markdown("#### 가능한 조치:")
                st.markdown("""
                - 이 값이 측정 오류인지 확인
                - 공정 로그를 검토하여 이 변수와 관련된 특이사항 확인
                - 유사한 특이값이 있었던 과거 배치와 비교
                """)
            else:
                st.success("✅ 이 배치의 모든 변수는 정상 범위 내에 있습니다.")
            
            # 결과 해석 도우미
            with st.expander("🔍 분석 결과 해석 도우미"):
                st.markdown("""
                ### Z-점수 해석 가이드
                
                1. **Z-점수 의미**
                   - Z-점수는 해당 값이 평균에서 얼마나 떨어져 있는지를 표준편차 단위로 나타냅니다.
                   - Z-점수 0 = 평균값
                   - Z-점수 +1 = 평균보다 1 표준편차 높음
                   - Z-점수 -1 = 평균보다 1 표준편차 낮음
                
                2. **특이값 판단 기준**
                   - |Z-점수| < 2: 일반적인 범위 (전체 데이터의 약 95%)
                   - 2 < |Z-점수| < 3: 약간 특이한 값 (전체 데이터의 약 4.5%)
                   - |Z-점수| > 3: 확실한 특이값 (전체 데이터의 약 0.3%)
                
                3. **그래프 해석**
                   - **점들의 분포**: 전체 배치의 해당 변수 값 분포
                   - **빨간점**: 선택한 배치의 해당 변수 값 위치
                   - **빨간 텍스트 상자**: 임계값을 초과한 Z-점수 (특이값)
                
                4. **조치 우선순위**
                   - Z-점수 절대값이 가장 큰 변수부터 조사
                   - 여러 변수가 동시에 특이값을 보이는 경우, 상호작용 관계 확인
                   - 특이 패턴(예: 여러 온도 변수가 모두 높음)이 있는지 확인
                """)
            
            # 상호작용 분석 (옵션)
            if len(outlier_vars) >= 2:
                st.markdown("### 특이 변수 간 상호작용 분석")
                
                with st.expander("❓ 특이 변수 간 상호작용을 분석하는 이유"):
                    st.markdown("""
                    ### 특이 변수 간 상관관계 분석의 중요성
                    
                    특이 변수들 간의 상관관계를 분석하는 것은 다음과 같은 중요한 의미가 있습니다:
                    
                    1. **근본 원인 파악**
                       - 여러 변수가 동시에 특이값을 보일 때, 이들 간의 관계를 파악하면 문제의 근본 원인을 찾는데 도움이 됩니다
                       - 예: 온도와 압력이 모두 특이값을 보이고 강한 양의 상관관계가 있다면, 하나의 제어로 두 문제를 해결할 수 있을 수 있습니다
                    
                    2. **연쇄 효과 이해**
                       - 한 변수의 이상이 다른 변수에 미치는 영향을 파악할 수 있습니다
                       - 예: 원료 투입량(특이값)과 제품 품질(특이값) 간에 강한 상관관계가 있다면, 원료 투입 공정 개선이 필요할 수 있습니다
                    
                    3. **개선 우선순위 설정**
                       - 여러 특이값이 발생했을 때, 상관관계가 높은 변수들을 우선적으로 관리하면 효율적인 품질 개선이 가능합니다
                       - 예: 여러 공정 변수 중 서로 강한 상관관계를 보이는 변수 그룹을 먼저 개선하면 연쇄적인 개선 효과를 기대할 수 있습니다
                    
                    4. **공정 최적화**
                       - 특이값을 보이는 변수들 간의 관계를 이해하면, 공정 조건을 최적화하는데 도움이 됩니다
                       - 예: 특정 공정 조건에서 여러 품질 특성이 동시에 특이값을 보인다면, 해당 조건의 조정이 필요할 수 있습니다
                    """)
                
                # 특이값 변수들만으로 상관관계 행렬 생성
                corr_matrix = data_analysis[outlier_vars].corr().round(3)
                
                # 상관관계 행렬 시각화와 테이블 나란히 표시 - 컬럼 비율 조정
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 삼각형 형태의 히트맵 생성 (대각선 위쪽만 표시)
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # 히트맵 생성 - 크기와 레이아웃 개선
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        text_auto='.2f',
                        title="특이 변수 간 상관관계",
                        labels=dict(color="상관계수"),
                        height=350,
                        width=400
                    )
                    
                    # 글씨 크기 증가 및 디자인 개선
                    fig.update_traces(
                        textfont=dict(size=15, color='black', family='Arial Black'),
                        texttemplate='%{text}'
                    )
                    
                    # 히트맵 레이아웃 조정
                    fig.update_layout(
                        coloraxis_colorbar=dict(
                            title="상관계수",
                            thicknessmode="pixels", thickness=15,
                            lenmode="pixels", len=250,
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=['-1', '-0.5', '0', '0.5', '1']
                        ),
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    # 대각선 위쪽만 표시하기 위해 마스크를 적용한 새로운 행렬 생성
                    masked_corr = corr_matrix.copy()
                    for i in range(len(corr_matrix)):
                        for j in range(i+1):
                            masked_corr.iloc[i, j] = None
                    
                    # 마스크된 데이터로 히트맵 업데이트
                    fig.data[0].z = masked_corr.values.tolist()
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 상관관계 테이블 (간결한 형태로)
                    st.markdown("#### 상관계수 값")
                    
                    # 상관관계가 있는 변수 쌍만 표시 (절댓값 0.1 이상)
                    pairs = []
                    for i in range(len(outlier_vars)):
                        for j in range(i+1, len(outlier_vars)):
                            var1, var2 = outlier_vars[i], outlier_vars[j]
                            corr_val = corr_matrix.loc[var1, var2]
                            if abs(corr_val) > 0.1:  # 의미 있는 상관관계만 표시
                                strength = "강함 💪" if abs(corr_val) > 0.7 else "중간 👌" if abs(corr_val) > 0.3 else "약함 👎"
                                pairs.append({
                                    "변수1": var1, 
                                    "변수2": var2, 
                                    "상관계수": corr_val,
                                    "강도": strength
                                })
                    
                    if pairs:
                        pair_df = pd.DataFrame(pairs)
                        st.dataframe(
                            pair_df.style.format({"상관계수": "{:.3f}"})
                            .background_gradient(cmap="RdBu_r", subset=["상관계수"])
                            .set_properties(**{'font-size': '15px'}),
                            use_container_width=True,
                            height=250
                        )
                        
                        # 해석 추가
                        strongest_pair = sorted(pairs, key=lambda x: abs(x["상관계수"]), reverse=True)[0]
                        st.markdown(f"""
                        **가장 강한 관계:** 
                        {strongest_pair['변수1']}와(과) {strongest_pair['변수2']} 사이의 상관계수는 
                        {strongest_pair['상관계수']:.3f}입니다.
                        """)
                    else:
                        st.info("의미 있는 상관관계(|r| > 0.1)가 없습니다.")

                # 상관관계 해석 정보 제공
                st.info("""
                **상관계수 해석:** • 1에 가까울수록: 강한 양의 상관관계 • -1에 가까울수록: 강한 음의 상관관계 • 0에 가까울수록: 상관관계 없음
                """)
                
                # 사용자가 세부 분석을 위한 변수 쌍 선택
                st.markdown("#### 세부 분석을 위한 변수 쌍 선택")
                
                # 변수 선택 컬럼 생성
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("첫 번째 변수:", outlier_vars, index=0)
                with col2:
                    var2 = st.selectbox("두 번째 변수:", 
                                      [v for v in outlier_vars if v != var1], 
                                      index=0)
                
                # 선택된 변수 쌍에 대한 산점도 생성 - 정사각형에 가깝게 조정
                fig = px.scatter(
                    data_analysis, 
                    x=var1, 
                    y=var2, 
                    title=f"{var1} vs {var2} 상호작용",
                    opacity=0.6,
                    height=450,
                    width=800
                )
                
                # 선택된 배치 포인트 추가
                fig.add_trace(
                    go.Scatter(
                        x=[data_analysis.loc[selected_batch, var1]],
                        y=[data_analysis.loc[selected_batch, var2]],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='circle'),
                        name=selected_batch
                    )
                )
                
                # 추세선 추가
                fig.update_layout(
                    margin=dict(l=20, r=20, t=60, b=40),
                    autosize=False
                )
                
                st.plotly_chart(fig, use_container_width=False)
    else:
        st.error("분석할 숫자형 변수가 없습니다.")
else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")


# 페이지 하단 소개
st.markdown("---")
st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ")