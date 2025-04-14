import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import shap
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
import traceback
from scipy import stats
import itertools

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파일 상단에 추가 (공통 그래프 설정)
def display_plot_centered(fig, width_pct=60):
    """그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.pyplot(fig)

def display_plotly_centered(fig, width_pct=60):
    """Plotly 그래프를 중앙에 표시하는 헬퍼 함수"""
    left_margin = (100 - width_pct) // 2
    right_margin = 100 - width_pct - left_margin
    cols = st.columns([left_margin, width_pct, right_margin])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=True)

st.title("4. 시뮬레이션")

# 시뮬레이션 개념 설명 추가
with st.expander("📚 예측 시뮬레이션이란?"):
    st.markdown("""
    ### 예측 시뮬레이션(Predictive Simulation)
    예측 시뮬레이션은 과거 데이터를 기반으로 학습된 모델을 사용하여 변수들의 다양한 값 조합에 따른 결과를 예측하는 과정입니다.
    
    ### 이 페이지의 기능
    
    **1. 데이터 준비**
    - CSV 파일을 업로드하여 분석 데이터를 준비합니다.
    - 예측하고자 하는 타겟 변수를 선택합니다.
    
    **2. 상관관계 분석**
    - 타겟 변수와 상관관계가 높은 주요 변수들을 식별합니다.
    - 이 변수들은 예측 모델에 중요한 인자가 됩니다.
    
    **3. 모델 훈련**
    - 머신러닝 알고리즘(RandomForest 또는 XGBoost)을 사용하여 예측 모델을 훈련합니다.
    - 훈련된 모델의 성능을 평가합니다(R² 및 RMSE).
    - 변수 중요도를 확인하여 모델에 영향을 주는 핵심 요인을 파악합니다.
    
    **4. 예측 시뮬레이션**
    - 주요 변수들의 값을 조정하면서 결과 변화를 즉시 확인합니다.
    - 다양한 시나리오에 따른 예측 결과를 비교할 수 있습니다.
    - SHAP 값을 통해 각 변수가 예측 결과에 기여하는 정도를 분석합니다.
    
    ### 활용 방법
    
    - **공정 최적화**: 원하는 결과를 얻기 위한 최적의 변수 조합을 찾습니다.
    - **민감도 분석**: 어떤 변수가 결과에 가장 큰 영향을 미치는지 파악합니다.
    - **품질 예측**: 특정 조건에서 제품 품질이 어떻게 변화할지 예측합니다.
    - **불량률 감소**: 불량 발생 가능성이 높은 조건을 사전에 식별하여 예방합니다.
    - **비용 절감**: 재료 및 에너지 소비를 최적화하여 비용을 절감합니다.
    
    ### 머신러닝 모델 설명
    
    **RandomForest(랜덤 포레스트)**
    - 여러 의사결정 트리의 앙상블 모델입니다.
    - 다양한 데이터 타입에 적용 가능하고 과적합에 강합니다.
    - 변수 간 상호작용을 잘 포착합니다.
    
    **XGBoost**
    - 그래디언트 부스팅 기반의 고성능 머신러닝 알고리즘입니다.
    - 일반적으로 더 높은 예측 정확도를 보이지만 튜닝이 필요할 수 있습니다.
    - 변수 중요도 및 해석이 가능합니다.
    """)

# 파일 업로드 영역
st.write("### 데이터 업로드")
uploaded_file = st.file_uploader("예측 모델링을 위한 CSV 파일을 업로드하세요", type=['csv'])

# 데이터 로드
data = None
if uploaded_file is not None:
    try:
        # 파일 내용 미리보기 (문제 진단용)
        file_bytes = uploaded_file.getvalue()
        try:
            preview_text = file_bytes[:1000].decode('utf-8')
        except UnicodeDecodeError:
            try:
                preview_text = file_bytes[:1000].decode('cp949')
            except UnicodeDecodeError:
                preview_text = "인코딩 문제로 미리보기를 표시할 수 없습니다."
        
        # 구분자 선택 옵션 추가
        st.write("#### 파일 미리보기:")
        st.text(preview_text)
        
        delimiter_option = st.selectbox(
            "CSV 구분자 선택:",
            options=[',', ';', '\t', '|'],
            index=0,
            format_func=lambda x: {',' : '쉼표(,)', ';' : '세미콜론(;)', '\t': '탭(\\t)', '|': '파이프(|)'}[x]
        )
        
        # 다양한 인코딩 시도
        for encoding in ['utf-8', 'euc-kr', 'cp949']:
            try:
                # 파일 포인터 위치 초기화
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file, encoding=encoding, delimiter=delimiter_option)
                if len(data.columns) <= 1 and data.columns[0].count(',') > 3:  # 구분자 감지 문제인 경우
                    st.warning("구분자 문제가 감지되었습니다. 다른 구분자를 선택해보세요.")
                    data = None
                    continue
                break
            except Exception as e:
                continue
        
        if data is not None and len(data.columns) > 1:
            st.success(f"파일 업로드 완료! 데이터 크기: {data.shape[0]}행 x {data.shape[1]}열")
            
            # 데이터 미리보기 표시
            st.write("#### 데이터 미리보기:")
            st.dataframe(data.head())
        else:
            st.error("파일을 읽는 데 문제가 발생했습니다. 다른 구분자를 선택하거나 파일 형식을 확인해주세요.")
            st.stop()
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()

if data is not None:
    # 타깃 변수 선택
    st.write("### 예측 타깃 변수 선택")
    
    # 수치형 변수만 추출
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("수치형 데이터 컬럼이 없습니다. 데이터 타입을 확인해주세요.")
        # 모든 컬럼을 문자열에서 숫자로 변환 시도
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
                numeric_cols.append(col)
            except:
                pass
        
        if not numeric_cols:
            st.error("변환 가능한 수치형 데이터가 없습니다. 다른 파일을 업로드해주세요.")
            st.stop()
    
    # 추천 타겟 자동 선택
    default_target = None
    for col in numeric_cols:
        if '용출' in col and ('최소' in col or 'min' in col.lower() or 'Min' in col):
            default_target = col
            break
    
    # 타겟 변수 선택 UI
    if default_target:
        st.info(f"'{default_target}' 컬럼이 기본 타겟으로 자동 선택되었습니다. 필요하면 변경하세요.")
    
    target_col = st.selectbox(
        "예측할 타깃 변수를 선택하세요:",
        numeric_cols,
        index=numeric_cols.index(default_target) if default_target and default_target in numeric_cols else 0
    )
    
    st.subheader(f"'{target_col}' 예측 모델링")
    
    # 데이터 전처리
    numeric_data = data.select_dtypes(include=[np.number])
    
    # 상관관계 분석
    if target_col in numeric_data.columns:
        # NaN 값 처리
        correlation_data = numeric_data.copy()
        correlation_data = correlation_data.fillna(correlation_data.mean())
        
        # 상관계수 계산 - 절대값을 사용하지 않고 원래 값을 유지
        correlations = correlation_data.corr()[target_col].sort_values(ascending=False)
        correlations = correlations.drop(target_col)  # 타깃 변수 자신과의 상관관계 제외
        
        # 상위 변수 선택 시 절대값으로 정렬하되, 표시할 때는 원래 값 사용
        top_indices = correlations.abs().sort_values(ascending=False).head(10).index
        correlation_with_target = correlations[top_indices]
        
        # 상관관계 시각화 (Plotly로 변경)
        fig_corr = go.Figure()
        
        # 상관계수에 따라 색상 설정
        colors = ['#3498db' if val > 0 else '#e74c3c' for val in correlation_with_target.values]
        
        # 가로 막대 차트 추가
        fig_corr.add_trace(
            go.Bar(
                y=correlation_with_target.index,
                x=correlation_with_target.values,  # 원래 값 사용 (음수/양수 유지)
                orientation='h',
                marker_color=colors,
                text=[f'{val:.2f}' for val in correlation_with_target.values],
                textposition='outside',
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            )
        )
        
        # 레이아웃 설정
        fig_corr.update_layout(
            title=f'{target_col}와(과)의 상관관계 (상위 10개)',
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
        
        # 탭 생성
        tab1, tab2 = st.tabs(["모델 훈련", "시뮬레이션"])
        
        with tab1:
            st.write("### 모델 훈련")
            
            # 회귀분석과 머신러닝의 차이점 설명
            with st.expander("💡 회귀분석과 머신러닝의 차이점 이해하기", expanded=True):
                st.markdown("""
                ### 회귀분석과 머신러닝의 차이점
                
                #### 1. 기본 개념
                - **회귀분석**: 변수 간의 관계를 수학적 방정식으로 표현하는 통계적 방법
                - **머신러닝**: 데이터로부터 패턴을 학습하여 예측하는 방법
                
                #### 2. 주요 차이점
                
                **📊 예측 방식**
                - **회귀분석**: 
                  - 선형 관계만 고려 (y = ax + b 형태)
                  - 변수 간 관계가 명확하고 해석 가능
                  - 이상치에 민감
                
                - **머신러닝**: 
                  - 비선형 관계도 학습 가능
                  - 복잡한 패턴 발견 가능
                  - 이상치에 더 강건함
                
                **🔍 해석성**
                - **회귀분석**: 
                  - 결과가 매우 명확하고 해석하기 쉬움
                  - 각 변수의 영향력을 정확히 파악 가능
                  - 통계적 유의성 검정 가능
                
                - **머신러닝**: 
                  - 결과 해석이 상대적으로 어려움
                  - '블랙박스'처럼 작동할 수 있음
                  - 변수 중요도는 파악 가능하나 정확한 영향력은 알기 어려움
                
                **🎯 적합한 상황**
                - **회귀분석이 좋은 경우**: 
                  - 변수 간 관계가 선형적일 때
                  - 결과의 해석이 중요할 때
                  - 데이터가 적을 때
                  - 통계적 검정이 필요할 때
                
                - **머신러닝이 좋은 경우**: 
                  - 복잡한 비선형 관계가 있을 때
                  - 예측 정확도가 가장 중요할 때
                  - 데이터가 많을 때
                  - 실시간 예측이 필요할 때
                
                #### 3. 실제 적용 시 고려사항
                - **데이터의 특성**: 
                  - 데이터가 적으면 회귀분석이 더 안정적
                  - 데이터가 많으면 머신러닝이 더 정확할 수 있음
                
                - **목적에 따른 선택**: 
                  - 해석이 중요하면 → 회귀분석
                  - 예측 정확도가 중요하면 → 머신러닝
                
                - **실제 사례**: 
                  - 품질 관리에서는 두 방법을 모두 사용
                  - 초기 분석에는 회귀분석으로 관계 파악
                  - 실제 예측에는 머신러닝 활용
                """)
            
            # 모델 선택
            st.write("### 모델 선택")
            
            # 모델 설명 추가
            with st.expander("💡 각 모델의 특징", expanded=True):
                st.markdown("""
                ### 모델 종류와 특징
                
                #### 1. RandomForest (랜덤 포레스트)
                - 여러 개의 의사결정 나무를 결합한 앙상블 모델
                - 안정적이고 과적합에 강함
                - 복잡한 관계도 잘 학습
                
                #### 2. XGBoost (엑스지부스트)
                - 가장 성능이 좋은 부스팅 알고리즘 중 하나
                - 높은 예측 정확도
                - 계산 속도가 빠름
                
                #### 3. 선형 회귀
                - 가장 기본적인 통계 모델
                - 결과 해석이 쉽고 직관적
                - 단순한 선형 관계에 적합
                """)
            
            model_type = st.radio(
                "모델 선택:",
                ["RandomForest", "XGBoost", "선형 회귀"],
                horizontal=True
            )
            
            # 데이터 전처리 옵션
            st.write("### 데이터 전처리 옵션")
            
            # 이상치 처리 옵션
            outlier_options = st.expander("이상치 처리 옵션", expanded=True)
            with outlier_options:
                remove_outliers = st.checkbox("이상치 제거", value=True, 
                                            help="학습 데이터에서 이상치를 제거하여 모델 성능을 향상시킵니다.")
                
                outlier_method = st.radio(
                    "이상치 탐지 방법:",
                    ["Z-점수", "IQR 방법"],
                    horizontal=True,
                    help="Z-점수: 평균에서 n 표준편차 이상 떨어진 값을 이상치로 간주\nIQR: 사분위수 범위를 벗어난 값을 이상치로 간주"
                )
                
                if outlier_method == "Z-점수":
                    z_threshold = st.slider("Z-점수 임계값", 
                                          min_value=2.0, max_value=5.0, value=3.0, step=0.1,
                                          help="이 값보다 큰 Z-점수를 가진 데이터를 이상치로 간주합니다.")
                else:  # IQR 방법
                    iqr_multiplier = st.slider("IQR 곱수", 
                                             min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                             help="IQR × 이 값을 초과하는 데이터를 이상치로 간주합니다.")
            
            # 데이터 스케일링 옵션
            scaling_options = st.expander("데이터 스케일링 옵션", expanded=True)
            with scaling_options:
                apply_scaling = st.checkbox("데이터 스케일링 적용", value=True,
                                          help="변수를 동일한 스케일로 조정하여 모델 성능을 향상시킵니다.")
                
                if apply_scaling:
                    scaler_method = st.radio(
                        "스케일링 방법:",
                        ["표준화(StandardScaler)", "로버스트 스케일링(RobustScaler)", "정규화(MinMaxScaler)"],
                        horizontal=True,
                        help="표준화: 평균 0, 표준편차 1로 변환\n로버스트: 중앙값과 IQR을 사용하여 이상치에 강건한 스케일링\n정규화: 최소-최대 스케일링으로 0~1 범위로 변환"
                    )
            
            # 훈련 버튼
            if st.button("모델 훈련"):
                with st.spinner("모델 훈련 중..."):
                    # 원본 데이터 보존
                    X_orig = correlation_data[top_indices].copy()
                    y_orig = correlation_data[target_col].copy()
                    
                    # 이상치 확인 및 시각화
                    if remove_outliers:
                        if outlier_method == "Z-점수":
                            # 이상치 제거 전 데이터 건수
                            before_count = len(X_orig)
                            
                            # Z-점수 계산 및 이상치 식별
                            z_scores = np.abs((X_orig - X_orig.mean()) / X_orig.std())
                            outlier_mask = (z_scores > z_threshold).any(axis=1)
                            
                            # 타겟 변수의 Z-점수도 계산
                            y_z_score = np.abs((y_orig - y_orig.mean()) / y_orig.std())
                            y_outlier_mask = (y_z_score > z_threshold)
                            
                            # 이상치가 있는 행을 식별
                            combined_mask = outlier_mask | y_outlier_mask
                            
                            # 이상치 제거된 데이터
                            X_cleaned = X_orig[~combined_mask]
                            y_cleaned = y_orig[~combined_mask]
                            
                            # 제거된 데이터 수 계산
                            removed_count = before_count - len(X_cleaned)
                            removal_percentage = (removed_count / before_count) * 100
                            
                            st.info(f"Z-점수 {z_threshold} 이상인 이상치 {removed_count}개({removal_percentage:.1f}%)를 제거했습니다. (원본: {before_count}개 → 정제: {len(X_cleaned)}개)")
                            
                        else:  # IQR 방법
                            # 이상치 제거 전 데이터 건수
                            before_count = len(X_orig)
                            
                            # 각 특성에 대해 IQR 기반 이상치 탐지
                            outlier_rows = []
                            for col in X_orig.columns:
                                Q1 = X_orig[col].quantile(0.25)
                                Q3 = X_orig[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                
                                # 이상치가 있는 행 인덱스
                                feature_outliers = X_orig[(X_orig[col] < lower_bound) | (X_orig[col] > upper_bound)].index
                                outlier_rows.extend(feature_outliers)
                            
                            # 타겟 변수의 이상치도 확인
                            Q1 = y_orig.quantile(0.25)
                            Q3 = y_orig.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            
                            target_outliers = y_orig[(y_orig < lower_bound) | (y_orig > upper_bound)].index
                            outlier_rows.extend(target_outliers)
                            
                            # 중복 제거하여 이상치가 있는 모든 행 인덱스
                            outlier_indices = list(set(outlier_rows))
                            
                            # 이상치 제거
                            X_cleaned = X_orig.drop(outlier_indices)
                            y_cleaned = y_orig.drop(outlier_indices)
                            
                            # 제거된 데이터 수 계산
                            removed_count = before_count - len(X_cleaned)
                            removal_percentage = (removed_count / before_count) * 100
                            
                            st.info(f"IQR × {iqr_multiplier} 범위를 벗어난 이상치 {removed_count}개({removal_percentage:.1f}%)를 제거했습니다. (원본: {before_count}개 → 정제: {len(X_cleaned)}개)")
                    else:
                        # 이상치 제거 없이 원본 데이터 사용
                        X_cleaned = X_orig.copy()
                        y_cleaned = y_orig.copy()
                    
                    # 데이터 부족 시 경고
                    if len(X_cleaned) < 20:
                        st.warning("데이터가 너무 적습니다(20개 미만). 모델 성능이 좋지 않을 수 있습니다.")
                        if remove_outliers:
                            st.warning("이상치 제거 기준을 완화하거나 비활성화하는 것을 고려하세요.")
                    
                    # 스케일링 적용
                    if apply_scaling:
                        if scaler_method == "표준화(StandardScaler)":
                            scaler = StandardScaler()
                            st.session_state.scaler = scaler
                            st.session_state.scaling_method = "standard"
                        elif scaler_method == "로버스트 스케일링(RobustScaler)":
                            scaler = RobustScaler()
                            st.session_state.scaler = scaler
                            st.session_state.scaling_method = "robust"
                        else:  # 정규화(MinMaxScaler)
                            scaler = MinMaxScaler()
                            st.session_state.scaler = scaler
                            st.session_state.scaling_method = "minmax"
                        
                        # X 데이터 스케일링
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X_cleaned),
                            columns=X_cleaned.columns,
                            index=X_cleaned.index
                        )
                        
                        st.info(f"{scaler_method}를 적용했습니다.")
                    else:
                        X_scaled = X_cleaned
                        st.session_state.scaling_method = None
                    
                    # 훈련/테스트 분할
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_cleaned, test_size=0.2, random_state=42
                    )
                    
                    # 모델 훈련
                    if model_type == "RandomForest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_type == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                    else:  # 선형 회귀
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                    
                    model.fit(X_train, y_train)
                    
                    # 모델 평가
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # 결과 출력
                    st.write(f"**평균 제곱 오차(MSE):** {mse:.4f}")
                    st.write(f"**평균 제곱근 오차(RMSE):** {rmse:.4f}")
                    st.write(f"**R² 점수:** {r2:.4f}")
                    
                    # 회귀 분석 결과 (선형 회귀 모델인 경우)
                    if model_type == "선형 회귀":
                        st.markdown("### 회귀 분석 결과")
                        
                        # 회귀 분석 해석 가이드를 먼저 표시
                        with st.expander("💡 회귀 분석 결과 쉽게 이해하기", expanded=True):
                            st.markdown("""
                            ### 회귀 분석 결과 쉽게 이해하기
                            
                            #### 1. 회귀 분석 결과 표 해석
                            - **회귀 계수**: 변수가 1 증가할 때 예측값이 얼마나 변하는지 나타내요
                              - **양수**: 이 변수가 1 증가하면 → 예측값도 증가해요
                              - **음수**: 이 변수가 1 증가하면 → 예측값은 감소해요
                              - **크기**: 숫자가 클수록 → 영향력이 커요
                            
                            - **표준 오차**: 회귀 계수의 불확실성을 나타내요
                              - **작을수록**: 계수가 더 정확하다는 의미예요
                              - **클수록**: 계수가 불확실하다는 의미예요
                            
                            - **t 통계량**: 회귀 계수가 0과 다른지 검정하는 값이에요
                              - **절대값이 클수록**: 변수가 더 중요하다는 의미예요
                              - **일반적으로 2 이상**: 변수가 중요하다고 볼 수 있어요
                            
                            - **p-value**: 변수가 통계적으로 의미 있는지 나타내요
                              - **0.05보다 작으면**: 이 변수가 정말 중요한 거예요! (통계적으로 의미 있어요)
                              - **0.05보다 크면**: 이 변수는 크게 중요하지 않아요
                            
                            #### 2. R² 점수는?
                            - **1에 가까울수록**: 모델이 정말 잘 예측하는 거예요
                            - **0에 가까울수록**: 모델이 잘 예측하지 못하는 거예요
                            - **일반적으로 0.7 이상**: 좋은 모델이라고 볼 수 있어요
                            
                            #### 3. 회귀 방정식 활용법
                            - 방정식을 보면 각 변수가 얼마나 영향을 주는지 알 수 있어요
                            - 예: 변수 A가 10이고 변수 B가 5일 때 예측값은?
                              1. 방정식에 숫자를 넣어서 계산하면 돼요
                              2. 양수 계수면 더하고, 음수 계수면 빼요
                            
                           
                            """)
                        
                        # 회귀 계수 및 p-value 계산
                        # 회귀 계수
                        coefficients = model.coef_
                        
                        # p-value 계산
                        n = len(X_train)
                        p = len(X_train.columns)
                        dof = n - p - 1
                        
                        # MSE 계산
                        mse = np.sum((y_train - model.predict(X_train)) ** 2) / dof
                        
                        # X의 공분산 행렬의 역행렬
                        X_with_intercept = np.column_stack([np.ones(n), X_train])
                        var_b = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
                        
                        # 표준 오차
                        sd_b = np.sqrt(var_b)
                        
                        # t 통계량
                        t_stat = coefficients / sd_b[1:]
                        
                        # p-value
                        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
                        
                        # 결과를 데이터프레임으로 변환
                        regression_results = pd.DataFrame({
                            '변수': X_train.columns,
                            '회귀 계수': coefficients,
                            '표준 오차': sd_b[1:],
                            't 통계량': t_stat,
                            'p-value': p_values
                        })
                        
                        # p-value 기준으로 정렬
                        regression_results = regression_results.sort_values('p-value')
                        
                        # 결과 표시
                        st.dataframe(
                            regression_results.style.format({
                                '회귀 계수': '{:.4f}',
                                '표준 오차': '{:.4f}',
                                't 통계량': '{:.4f}',
                                'p-value': '{:.4f}'
                            }).background_gradient(cmap='RdYlBu_r', subset=['p-value']),
                            use_container_width=True
                        )
                        
                        # 회귀 방정식 표시
                        st.markdown("#### 회귀 방정식:")
                        equation = f"{target_col} = {model.intercept_:.4f}"
                        for i, coef in enumerate(coefficients):
                            if coef >= 0:
                                equation += f" + {coef:.4f} × {X_train.columns[i]}"
                            else:
                                equation += f" - {abs(coef):.4f} × {X_train.columns[i]}"
                        st.markdown(f"**{equation}**")
                        
                        # 회귀 모델 가정 검정
                        st.markdown("#### 회귀 모델 가정 검정")
                        
                        # 회귀 모델 가정 검정 설명
                        with st.expander("💡 회귀 모델 가정 검정 이해하기", expanded=True):
                            st.markdown("""
                            ### 회귀 모델 가정 검정 이해하기
                            
                            회귀 분석은 세 가지 주요 가정을 만족해야 신뢰할 수 있는 결과를 얻을 수 있습니다:
                            
                            1. **정규성(Normality)**
                               - 잔차(예측값과 실제값의 차이)가 정규분포를 따라야 합니다
                               - 이는 통계적 추론의 유효성을 보장합니다
                            
                            2. **선형성(Linearity)**
                               - 예측변수와 반응변수 간의 관계가 선형적이어야 합니다
                               - 잔차가 예측값에 대해 무작위로 분포해야 합니다
                            
                            3. **등분산성(Homoscedasticity)**
                               - 잔차의 분산이 모든 예측값에서 동일해야 합니다
                               - 이는 모델의 예측이 모든 범위에서 동일한 정확도를 가져야 함을 의미합니다
                            
                            아래 그래프들을 통해 이러한 가정들이 만족되는지 확인할 수 있습니다.
                            """)
                        
                        # 1. 정규성 검정 (잔차의 정규성)
                        residuals = y_train - model.predict(X_train)
                        _, p_value = stats.normaltest(residuals)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # 정규성 검정 결과
                            st.markdown("**1. 잔차의 정규성 검정**")
                            if p_value < 0.05:
                                st.warning(f"잔차가 정규분포를 따르지 않습니다 (p-value: {p_value:.4f})")
                            else:
                                st.success(f"잔차가 정규분포를 따릅니다 (p-value: {p_value:.4f})")
                            
                            # 정규성 검정 설명
                            st.markdown("""
                            #### 정규성 검정 그래프 해석
                            - 이 그래프는 잔차의 분포를 보여줍니다
                            - 이상적인 경우: 종 모양(bell-shaped)의 대칭적인 분포
                            - 빨간색 점선: 정규분포 곡선
                            - 해석:
                              - 분포가 대칭적이고 종 모양이면 → 정규성 가정 만족
                              - 분포가 비대칭이거나 꼬리가 두꺼우면 → 정규성 가정 위반
                            """)
                            
                            # 잔차 히스토그램
                            fig_residuals = go.Figure()
                            fig_residuals.add_trace(
                                go.Histogram(
                                    x=residuals,
                                    nbinsx=30,
                                    name='잔차',
                                    marker_color='#3498db'
                                )
                            )
                            
                            # 정규분포 곡선 추가
                            x_range = np.linspace(min(residuals), max(residuals), 100)
                            y_range = stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals))
                            fig_residuals.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=y_range,
                                    mode='lines',
                                    name='정규분포',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            
                            fig_residuals.update_layout(
                                title='잔차 분포',
                                xaxis_title='잔차',
                                yaxis_title='빈도',
                                height=300
                            )
                            st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        with col2:
                            # 2. 선형성 검정 (잔차 vs 예측값)
                            st.markdown("**2. 선형성 검정**")
                            
                            # 선형성 검정 설명
                            st.markdown("""
                            #### 선형성 검정 그래프 해석
                            - 이 그래프는 예측값과 잔차의 관계를 보여줍니다
                            - 이상적인 경우: 점들이 무작위로 분포하고 패턴이 없어야 함
                            - 빨간색 점선: 0선 (잔차가 0인 기준선)
                            - 해석:
                              - 점들이 무작위로 분포하면 → 선형성 가정 만족
                              - 점들이 패턴을 보이면 → 선형성 가정 위반
                              - 곡선형 패턴이 보이면 → 비선형 관계 존재
                            """)
                            
                            # 잔차 vs 예측값 산점도
                            fig_linearity = go.Figure()
                            fig_linearity.add_trace(
                                go.Scatter(
                                    x=model.predict(X_train),
                                    y=residuals,
                                    mode='markers',
                                    marker=dict(color='#3498db', size=8, opacity=0.6),
                                    name='잔차 vs 예측값'
                                )
                            )
                            
                            # 0선 추가
                            fig_linearity.add_shape(
                                type="line",
                                x0=min(model.predict(X_train)),
                                y0=0,
                                x1=max(model.predict(X_train)),
                                y1=0,
                                line=dict(color="red", width=1, dash="dash")
                            )
                            
                            fig_linearity.update_layout(
                                title='잔차 vs 예측값',
                                xaxis_title='예측값',
                                yaxis_title='잔차',
                                height=300
                            )
                            st.plotly_chart(fig_linearity, use_container_width=True)
                        
                        # 3. 등분산성 검정
                        st.markdown("**3. 등분산성 검정**")
                        
                        # 등분산성 검정 설명
                        st.markdown("""
                        #### 등분산성 검정 그래프 해석
                        - 이 그래프는 예측값과 잔차의 절대값 관계를 보여줍니다
                        - 이상적인 경우: 점들이 무작위로 분포하고 패턴이 없어야 함
                        - 해석:
                          - 점들이 무작위로 분포하면 → 등분산성 가정 만족
                          - 점들이 깔때기 모양이나 다른 패턴을 보이면 → 등분산성 가정 위반
                          - 잔차의 크기가 예측값에 따라 변면 → 이분산성(heteroscedasticity) 문제
                        """)
                        
                        # 잔차의 절대값 vs 예측값
                        fig_homoscedasticity = go.Figure()
                        fig_homoscedasticity.add_trace(
                            go.Scatter(
                                x=model.predict(X_train),
                                y=np.abs(residuals),
                                mode='markers',
                                marker=dict(color='#3498db', size=8, opacity=0.6),
                                name='|잔차| vs 예측값'
                            )
                        )
                        
                        fig_homoscedasticity.update_layout(
                            title='|잔차| vs 예측값 (등분산성 검정)',
                            xaxis_title='예측값',
                            yaxis_title='|잔차|',
                            height=300
                        )
                        st.plotly_chart(fig_homoscedasticity, use_container_width=True)
                    
                    # 모델 및 특성 저장
                    st.session_state.model = model
                    st.session_state.model_features = top_indices.tolist()
                    st.session_state.remove_outliers = remove_outliers
                    st.session_state.apply_scaling = apply_scaling
                    st.session_state.model_type = model_type
                    
                    if remove_outliers:
                        if outlier_method == "Z-점수":
                            st.session_state.outlier_method = "zscore"
                            st.session_state.outlier_param = z_threshold
                        else:
                            st.session_state.outlier_method = "iqr"
                            st.session_state.outlier_param = iqr_multiplier
                    
                    # 실제값-예측값 비교 그래프 (Plotly로 변경)
                    fig_compare = go.Figure()

                    # 산점도 추가
                    fig_compare.add_trace(
                        go.Scatter(
                            x=y_test,
                            y=y_pred,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color='rgba(0, 123, 255, 0.7)',
                                line=dict(
                                    color='rgba(0, 123, 255, 1.0)',
                                    width=1
                                )
                            ),
                            name='예측값',
                            hovertemplate='실제값: %{x:.4f}<br>예측값: %{y:.4f}<extra></extra>'
                        )
                    )

                    # 이상적인 예측선 (대각선) 추가
                    min_val = min(min(y_test), min(y_pred))
                    max_val = max(max(y_test), max(y_pred))
                    fig_compare.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='이상적인 예측'
                        )
                    )

                    # 레이아웃 설정
                    fig_compare.update_layout(
                        title='실제값 vs 예측값',
                        xaxis_title='실제값',
                        yaxis_title='예측값',
                        height=500,
                        width=700,
                        showlegend=True,
                        hovermode='closest'
                    )

                    # 축 범위를 동일하게 설정
                    overall_min = min(min_val, min_val)
                    overall_max = max(max_val, max_val)
                    padding = (overall_max - overall_min) * 0.05
                    fig_compare.update_xaxes(range=[overall_min - padding, overall_max + padding])
                    fig_compare.update_yaxes(range=[overall_min - padding, overall_max + padding])

                    # 그리드 추가
                    fig_compare.update_layout(
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGrey'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGrey'
                        )
                    )

                    # 그래프 표시
                    display_plotly_centered(fig_compare)
                    
                    # 변수 중요도 시각화
                    if model_type in ["RandomForest", "XGBoost"]:
                        # 랜덤 포레스트와 XGBoost의 변수 중요도
                        feature_importance = model.feature_importances_
                        
                        # 변수 중요도를 데이터프레임으로 변환
                        feature_importance_df = pd.DataFrame({
                            'Feature': X_train.columns,
                            'Importance': feature_importance
                        })

                        # 중요도 기준 내림차순 정렬
                        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

                        # Plotly 그래프 생성
                        fig_importance = go.Figure()

                        # 바 차트 추가 (남색으로 변경)
                        fig_importance.add_trace(
                            go.Bar(
                                y=feature_importance_df['Feature'],
                                x=feature_importance_df['Importance'],
                                orientation='h',
                                marker_color='#3498db',  # 남색으로 변경
                                text=[f'{val:.4f}' for val in feature_importance_df['Importance']],
                                textposition='outside',
                                hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                            )
                        )

                        # 레이아웃 설정
                        fig_importance.update_layout(
                            title='모델의 전반적인 변수 중요도',
                            xaxis_title='중요도',
                            yaxis_title='변수',
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis=dict(
                                range=[0, max(feature_importance_df['Importance']) * 1.1],  # 0부터 시작하도록 수정
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='LightGrey'
                            ),
                            yaxis=dict(
                                autorange='reversed'  # 중요도가 높은 순으로 정렬
                            )
                        )

                        # 그래프 표시
                        st.subheader("모델의 전반적인 변수 중요도")
                        display_plotly_centered(fig_importance)
                    
                    # 모델 훈련 완료 메시지와 시뮬레이션 버튼
                    st.success("모델 훈련이 완료되었습니다!")
        
        with tab2:
            st.write("### 시뮬레이션")
            
            if 'model' in st.session_state and 'model_features' in st.session_state:
                # 시뮬레이션 모드 선택
                simulation_mode = st.radio(
                    "시뮬레이션 모드:",
                    ["수동 시뮬레이션", "최적화 시뮬레이션"],
                    horizontal=True
                )
                
                if simulation_mode == "수동 시뮬레이션":
                    st.write("아래 변수들의 값을 조정하여 예측해보세요:")
                    
                    # 입력 위젯 생성
                    input_values = {}
                    for feature in st.session_state.model_features:
                        min_val = float(numeric_data[feature].min())
                        max_val = float(numeric_data[feature].max())
                        mean_val = float(numeric_data[feature].mean())
                        std_val = float(numeric_data[feature].std())
                        
                        # 슬라이더 생성
                        input_values[feature] = st.slider(
                            f"{feature} (평균: {mean_val:.2f}, 표준편차: {std_val:.2f})",
                            min_val,
                            max_val,
                            mean_val,
                            step=(max_val-min_val)/100
                        )
                    
                    # 예측 수행 버튼
                    if st.button("예측 수행"):
                        with st.spinner("예측 중..."):
                            # 입력값으로 데이터프레임 생성
                            input_df = pd.DataFrame([input_values])
                            
                            # 스케일링 적용 (필요한 경우)
                            if 'apply_scaling' in st.session_state and st.session_state.apply_scaling:
                                input_scaled = st.session_state.scaler.transform(input_df)
                                input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
                                prediction = st.session_state.model.predict(input_scaled_df)[0]
                            else:
                                prediction = st.session_state.model.predict(input_df)[0]
                            
                            # 예측 결과를 세션 상태에 저장 (지속성 유지)
                            st.session_state.last_prediction = prediction
                            st.session_state.last_input_values = input_values.copy()
                            
                            # 타겟 통계 정보 저장
                            st.session_state.target_mean = numeric_data[target_col].mean()
                            st.session_state.target_min = numeric_data[target_col].min()
                            st.session_state.target_max = numeric_data[target_col].max()
                
                else:  # 최적화 시뮬레이션
                    st.write("### 최적화 시뮬레이션")
                    st.write("목표값을 설정하고 최적의 변수 조합을 찾아보세요.")
                    
                    # 목표값 설정
                    target_value = st.number_input(
                        f"목표 {target_col} 값:",
                        min_value=float(numeric_data[target_col].min()),
                        max_value=float(numeric_data[target_col].max()),
                        value=float(numeric_data[target_col].mean()),
                        step=0.1
                    )
                    
                    # 최적화 방법 선택
                    optimization_method = st.radio(
                        "최적화 방법:",
                        ["랜덤 서치"],
                        horizontal=True
                    )
                    
                    # 랜덤 서치 설명 추가
                    with st.expander("💡 랜덤 서치(Random Search)란?", expanded=True):
                        st.markdown("""
                        ### 랜덤 서치(Random Search) 이해하기
                        
                        랜덤 서치는 최적화 문제를 해결하기 위한 효율적인 방법입니다:
                        
                        #### 1. 기본 개념
                        - **랜덤 서치**: 변수의 가능한 값 범위 내에서 무작위로 값을 선택하여 최적의 조합을 찾는 방법
                        - **장점**: 
                          - 그리드 서치보다 훨씬 빠른 속도
                          - 더 넓은 탐색 범위 커버
                          - 지역 최적해에 덜 민감
                        
                        #### 2. 작동 방식
                        1. 각 변수에 대해 설정된 범위 내에서 무작위로 값을 선택
                        2. 선택된 값들로 예측을 수행
                        3. 목표값과 가장 가까운 결과를 찾을 때까지 반복
                        
                        #### 3. 그리드 서치와의 차이점
                        - **그리드 서치**: 모든 가능한 조합을 체계적으로 시도 (느림)
                        - **랜덤 서치**: 무작위로 선택된 조합만 시도 (빠름)
                        
                        #### 4. 활용 시 고려사항
                        - 시도 횟수를 늘리면 더 좋은 결과를 얻을 수 있음
                        - 변수의 범위를 적절히 설정하는 것이 중요
                        - 목표값에 도달하지 못할 경우 범위 조정 필요
                        """)
                    
                    # 최적화 범위 설정
                    st.write("#### 변수 범위 설정")
                    variable_ranges = {}
                    
                    for feature in st.session_state.model_features:
                        min_val = float(numeric_data[feature].min())
                        max_val = float(numeric_data[feature].max())
                        mean_val = float(numeric_data[feature].mean())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            variable_ranges[feature] = {
                                'min': st.number_input(
                                    f"{feature} 최소값:",
                                    min_value=min_val,
                                    max_value=mean_val,
                                    value=min_val,
                                    step=(mean_val-min_val)/20
                                )
                            }
                        with col2:
                            variable_ranges[feature]['max'] = st.number_input(
                                f"{feature} 최대값:",
                                min_value=mean_val,
                                max_value=max_val,
                                value=max_val,
                                step=(max_val-mean_val)/20
                            )
                    
                    # 최적화 버튼
                    if st.button("최적화 수행"):
                        with st.spinner("최적화 중..."):
                            # 최적화 수행
                            best_input_values = {}
                            best_prediction = None
                            min_diff = float('inf')
                            
                            # 랜덤 서치 파라미터
                            n_iterations = 1000  # 시도할 조합의 수
                            
                            # 진행 상황 표시
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 랜덤 서치 수행
                            for i in range(n_iterations):
                                # 진행 상황 업데이트
                                progress = (i + 1) / n_iterations
                                progress_bar.progress(progress)
                                status_text.text(f"진행 중: {i+1}/{n_iterations} 조합 시도 중...")
                                
                                # 랜덤 입력값 생성
                                current_input = {}
                                for feature in st.session_state.model_features:
                                    min_val = variable_ranges[feature]['min']
                                    max_val = variable_ranges[feature]['max']
                                    current_input[feature] = np.random.uniform(min_val, max_val)
                                
                                input_df = pd.DataFrame([current_input])
                                
                                # 스케일링 적용 (필요한 경우)
                                if 'apply_scaling' in st.session_state and st.session_state.apply_scaling:
                                    input_scaled = st.session_state.scaler.transform(input_df)
                                    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
                                    prediction = st.session_state.model.predict(input_scaled_df)[0]
                                else:
                                    prediction = st.session_state.model.predict(input_df)[0]
                                
                                # 목표값과의 차이 계산
                                diff = abs(prediction - target_value)
                                
                                # 최적 조합 업데이트
                                if diff < min_diff:
                                    min_diff = diff
                                    best_prediction = prediction
                                    best_input_values = current_input.copy()
                            
                            # 진행 상황 업데이트 완료
                            progress_bar.progress(1.0)
                            status_text.text("최적화 완료!")
                            
                            # 최적화 결과 저장
                            st.session_state.last_prediction = best_prediction
                            st.session_state.last_input_values = best_input_values
                            
                            # 타겟 통계 정보 저장
                            st.session_state.target_mean = numeric_data[target_col].mean()
                            st.session_state.target_min = numeric_data[target_col].min()
                            st.session_state.target_max = numeric_data[target_col].max()
                
                # 예측 결과가 있으면 항상 표시 (변수별 기여도 분석과 무관하게 유지)
                if 'last_prediction' in st.session_state:
                    prediction = st.session_state.last_prediction
                    target_mean = st.session_state.target_mean
                    target_min = st.session_state.target_min
                    target_max = st.session_state.target_max
                    
                    # 예측 결과 표시
                    st.success(f"### 예측된 {target_col}: {prediction:.4f}")
                    
                    # 추가 정보: 평균 및 범위와 비교
                    col1, col2, col3 = st.columns(3)
                    col1.metric("평균 대비", f"{(prediction - target_mean):.4f}", 
                                f"{((prediction - target_mean) / target_mean * 100):.2f}%")
                    col2.metric("최소값", f"{target_min:.4f}")
                    col3.metric("최대값", f"{target_max:.4f}")
                    
                    # 최적화 모드인 경우 목표값과의 차이 표시
                    if simulation_mode == "최적화 시뮬레이션":
                        st.metric("목표값과의 차이", f"{abs(prediction - target_value):.4f}")
                    
                    # 최적 변수 값 표시
                    st.write("#### 최적 변수 값:")
                    optimal_values_df = pd.DataFrame({
                        '변수': list(st.session_state.last_input_values.keys()),
                        '값': list(st.session_state.last_input_values.values())
                    })
                    st.dataframe(optimal_values_df, use_container_width=True)
            else:
                st.info("먼저 모델을 훈련해주세요.")
    else:
        st.error(f"타깃 변수 '{target_col}'을 찾을 수 없습니다.")
else:
    st.info("시뮬레이션을 시작하려면 CSV 파일을 업로드해주세요.") 