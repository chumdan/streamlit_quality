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
    - 이상치 제거와 데이터 스케일링 옵션을 제공합니다.
    
    **2. 상관관계 분석**
    - 타겟 변수와 상관관계가 높은 주요 변수들을 식별합니다.
    - 이 변수들은 예측 모델에 중요한 인자가 됩니다.
    - 상관관계 시각화를 통해 변수 간 관계를 쉽게 파악할 수 있습니다.
    
    **3. 모델 훈련**
    - 세 가지 모델 옵션 제공:
      - **선형 회귀**: 변수 간 선형 관계를 분석하고 해석이 용이합니다.
      - **RandomForest**: 비선형 관계도 포착하며 과적합에 강합니다.
      - **XGBoost**: 높은 예측 정확도를 제공하는 고성능 알고리즘입니다.
    - 모델 성능 평가(R², RMSE) 및 변수 중요도 분석을 제공합니다.
    - 회귀 모델의 경우 상세한 통계 분석 결과를 제공합니다.
    
    **4. 예측 시뮬레이션**
    두 가지 시뮬레이션 모드를 제공합니다:
    
    **A. 수동 시뮬레이션**
    - 주요 변수들의 값을 직접 조정하며 결과 변화를 확인
    - 각 변수의 평균과 표준편차 정보 제공
    - 실시간으로 예측 결과 확인 가능
    
    **B. 최적화 시뮬레이션**
    - 목표값을 설정하고 자동으로 최적의 변수 조합을 탐색
    - 랜덤 서치 알고리즘을 통한 효율적인 최적화
    - 변수별 탐색 범위 설정 가능
    - 최적화 진행 상황을 실시간으로 확인
    
    ### 활용 방법
    
    - **공정 최적화**: 원하는 결과를 얻기 위한 최적의 변수 조합을 찾습니다.
    - **민감도 분석**: 어떤 변수가 결과에 가장 큰 영향을 미치는지 파악합니다.
    - **품질 예측**: 특정 조건에서 제품 품질이 어떻게 변화할지 예측합니다.
    - **불량률 감소**: 불량 발생 가능성이 높은 조건을 사전에 식별하여 예방합니다.
    - **비용 절감**: 재료 및 에너지 소비를 최적화하여 비용을 절감합니다.
    - **가설 검증**: 특정 변수 조정이 결과에 미치는 영향을 사전 검증합니다.
    
    ### 데이터 전처리 옵션
    
    **1. 이상치 처리**
    - Z-점수 또는 IQR 방법을 통한 이상치 제거
    - 이상치 제거 기준 조정 가능
    
    **2. 데이터 스케일링**
    - 표준화(StandardScaler): 평균 0, 표준편차 1로 변환
    - 로버스트 스케일링(RobustScaler): 이상치에 강건한 스케일링
    - 정규화(MinMaxScaler): 0~1 범위로 변환
    
    ### 머신러닝 모델 설명
    
    **1. 선형 회귀**
    - 변수 간 선형 관계를 모델링
    - 회귀 계수를 통한 직관적인 해석 가능
    - 정규성, 선형성, 등분산성 검정 제공
    
    **2. RandomForest(랜덤 포레스트)**
    - 여러 의사결정 트리의 앙상블 모델
    - 다양한 데이터 타입에 적용 가능
    - 과적합에 강하고 변수 간 상호작용을 잘 포착
    
    **3. XGBoost**
    - 그래디언트 부스팅 기반의 고성능 알고리즘
    - 일반적으로 더 높은 예측 정확도
    - 변수 중요도 및 해석 가능
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
    st.write("### 변수 선택")
    
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
    
    # 1. 원인변수 선택
    st.write("#### 1. 원인변수 선택")
    selected_features = st.multiselect(
        "원인변수를 선택하세요 (여러 개 선택 가능):",
        options=numeric_cols,
        help="예측에 사용할 원인변수들을 선택하세요."
    )
    
    # 2. 결과변수 선택
    st.write("#### 2. 결과변수 선택")
    # 원인변수로 선택되지 않은 변수들 중에서 결과변수 선택
    remaining_cols = [col for col in numeric_cols if col not in selected_features]
    target_col = st.selectbox(
        "결과변수를 선택하세요:",
        options=remaining_cols,
        help="예측하고자 하는 결과변수를 선택하세요."
    )
    
    if selected_features and target_col:
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
            with st.expander("💡 회귀분석과 머신러닝의 차이점 이해하기", expanded=False):
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
            with st.expander("💡 각 모델의 특징", expanded=False):
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
            
            # 하이퍼파라미터 튜닝 옵션 추가
            tune_hyperparams = st.checkbox("하이퍼파라미터 튜닝 적용", value=False,
                                          help="모델의 하이퍼파라미터를 자동으로 최적화합니다.")
            
            # 데이터 전처리 옵션
            st.write("### 데이터 전처리 옵션")
            
            # 데이터 증강 옵션 추가
            data_augmentation_options = st.expander("데이터 증강 옵션", expanded=False)
            with data_augmentation_options:
                apply_augmentation = st.checkbox("데이터 증강 적용", value=False,
                                               help="데이터 증강을 통해 학습 데이터를 늘립니다.")
                
                if apply_augmentation:
                    augmentation_method = st.radio(
                        "증강 방법:",
                        ["SMOTE", "가우시안 노이즈", "선형 보간"],
                        horizontal=True,
                        help="SMOTE: 소수 클래스의 샘플을 보간하여 증강\n가우시안 노이즈: 기존 데이터에 노이즈를 추가\n선형 보간: 기존 데이터 포인트 사이를 보간"
                    )
                    
                    if augmentation_method == "SMOTE":
                        # SMOTE는 분류 문제에 주로 사용되므로, 회귀 문제에 맞게 수정
                        st.info("SMOTE는 분류 문제에 주로 사용되지만, 회귀 문제에도 적용할 수 있습니다.")
                        smote_samples = st.slider("생성할 샘플 수", 
                                                min_value=10, max_value=100, value=50, step=10,
                                                help="생성할 샘플 수를 선택하세요. 너무 많은 샘플은 과적합을 유발할 수 있습니다.")
                    
                    elif augmentation_method == "가우시안 노이즈":
                        noise_level = st.slider("노이즈 수준", 
                                              min_value=0.01, max_value=0.1, value=0.05, step=0.01,
                                              help="추가할 노이즈의 표준편차를 선택하세요.")
                        noise_samples = st.slider("생성할 샘플 수", 
                                                min_value=10, max_value=100, value=50, step=10,
                                                help="생성할 샘플 수를 선택하세요.")
                    
                    else:  # 선형 보간
                        interpolation_samples = st.slider("보간 샘플 수", 
                                                        min_value=10, max_value=100, value=50, step=10,
                                                        help="기존 데이터 포인트 사이에 생성할 샘플 수를 선택하세요.")
            
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
                    
                    # 데이터 증강 적용
                    if apply_augmentation:
                        # 데이터 증강 전 데이터 건수
                        before_aug_count = len(X_cleaned)
                        
                        # 데이터 증강 방법에 따라 적용
                        if augmentation_method == "SMOTE":
                            # SMOTE는 분류 문제에 주로 사용되므로, 회귀 문제에 맞게 수정
                            from sklearn.neighbors import NearestNeighbors
                            
                            # 데이터 증강을 위한 함수
                            def smote_for_regression(X, y, n_samples):
                                # 데이터 증강을 위한 결과 저장
                                X_aug = X.copy()
                                y_aug = y.copy()
                                
                                # 타겟 변수를 구간으로 나누어 각 구간별로 샘플링
                                y_bins = pd.qcut(y, q=5, labels=False)
                                
                                # 각 구간별로 SMOTE 적용
                                for i in range(5):
                                    # 현재 구간의 인덱스
                                    current_indices = np.where(y_bins == i)[0]
                                    
                                    if len(current_indices) < 2:
                                        continue
                                    
                                    # 현재 구간의 데이터
                                    X_current = X.iloc[current_indices]
                                    y_current = y.iloc[current_indices]
                                    
                                    # k-최근접 이웃 찾기
                                    nbrs = NearestNeighbors(n_neighbors=2).fit(X_current)
                                    distances, indices = nbrs.kneighbors(X_current)
                                    
                                    # 각 샘플에 대해 보간
                                    for j in range(min(n_samples // 5, len(current_indices))):
                                        # 랜덤하게 두 이웃 선택
                                        idx = np.random.randint(0, len(current_indices))
                                        neighbor_idx = indices[idx, 1]
                                        
                                        # 보간 계수
                                        alpha = np.random.random()
                                        
                                        # 보간된 샘플 생성
                                        X_interp = X_current.iloc[idx] * (1 - alpha) + X_current.iloc[neighbor_idx] * alpha
                                        y_interp = y_current.iloc[idx] * (1 - alpha) + y_current.iloc[neighbor_idx] * alpha
                                        
                                        # 증강된 데이터 추가
                                        X_aug = pd.concat([X_aug, pd.DataFrame([X_interp], columns=X.columns)], ignore_index=True)
                                        y_aug = pd.concat([y_aug, pd.Series([y_interp], name=y.name)], ignore_index=True)
                                
                                return X_aug, y_aug
                            
                            # SMOTE 적용
                            X_aug, y_aug = smote_for_regression(X_cleaned, y_cleaned, smote_samples)
                            
                            # 증강된 데이터 수 계산
                            aug_count = len(X_aug) - before_aug_count
                            aug_percentage = (aug_count / before_aug_count) * 100
                            
                            st.success(f"SMOTE를 통해 {aug_count}개({aug_percentage:.1f}%)의 샘플을 증강했습니다. (원본: {before_aug_count}개 → 증강: {len(X_aug)}개)")
                            
                            # 증강된 데이터 사용
                            X_cleaned = X_aug
                            y_cleaned = y_aug
                            
                        elif augmentation_method == "가우시안 노이즈":
                            # 가우시안 노이즈 추가
                            X_aug = X_cleaned.copy()
                            y_aug = y_cleaned.copy()
                            
                            for _ in range(noise_samples):
                                # 랜덤하게 원본 데이터 선택
                                idx = np.random.randint(0, len(X_cleaned))
                                
                                # 가우시안 노이즈 생성
                                X_noise = X_cleaned.iloc[idx] + np.random.normal(0, noise_level, size=len(X_cleaned.columns))
                                y_noise = y_cleaned.iloc[idx] + np.random.normal(0, noise_level)
                                
                                # 증강된 데이터 추가
                                X_aug = pd.concat([X_aug, pd.DataFrame([X_noise], columns=X_cleaned.columns)], ignore_index=True)
                                y_aug = pd.concat([y_aug, pd.Series([y_noise], name=y_cleaned.name)], ignore_index=True)
                            
                            # 증강된 데이터 수 계산
                            aug_count = len(X_aug) - before_aug_count
                            aug_percentage = (aug_count / before_aug_count) * 100
                            
                            st.success(f"가우시안 노이즈를 통해 {aug_count}개({aug_percentage:.1f}%)의 샘플을 증강했습니다. (원본: {before_aug_count}개 → 증강: {len(X_aug)}개)")
                            
                            # 증강된 데이터 사용
                            X_cleaned = X_aug
                            y_cleaned = y_aug
                            
                        else:  # 선형 보간
                            # 선형 보간
                            X_aug = X_cleaned.copy()
                            y_aug = y_cleaned.copy()
                            
                            # 데이터 포인트 쌍 생성
                            pairs = []
                            for i in range(len(X_cleaned)):
                                for j in range(i+1, len(X_cleaned)):
                                    pairs.append((i, j))
                            
                            # 랜덤하게 쌍 선택
                            if len(pairs) > interpolation_samples:
                                selected_pairs = np.random.choice(len(pairs), interpolation_samples, replace=False)
                            else:
                                selected_pairs = np.arange(len(pairs))
                            
                            # 선택된 쌍에 대해 보간
                            for pair_idx in selected_pairs:
                                i, j = pairs[pair_idx]
                                
                                # 보간 계수
                                alpha = np.random.random()
                                
                                # 보간된 샘플 생성
                                X_interp = X_cleaned.iloc[i] * (1 - alpha) + X_cleaned.iloc[j] * alpha
                                y_interp = y_cleaned.iloc[i] * (1 - alpha) + y_cleaned.iloc[j] * alpha
                                
                                # 증강된 데이터 추가
                                X_aug = pd.concat([X_aug, pd.DataFrame([X_interp], columns=X_cleaned.columns)], ignore_index=True)
                                y_aug = pd.concat([y_aug, pd.Series([y_interp], name=y_cleaned.name)], ignore_index=True)
                            
                            # 증강된 데이터 수 계산
                            aug_count = len(X_aug) - before_aug_count
                            aug_percentage = (aug_count / before_aug_count) * 100
                            
                            st.success(f"선형 보간을 통해 {aug_count}개({aug_percentage:.1f}%)의 샘플을 증강했습니다. (원본: {before_aug_count}개 → 증강: {len(X_aug)}개)")
                            
                            # 증강된 데이터 사용
                            X_cleaned = X_aug
                            y_cleaned = y_aug
                    
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
                        if tune_hyperparams:
                            # 하이퍼파라미터 튜닝
                            from sklearn.model_selection import RandomizedSearchCV
                            
                            # 파라미터 그리드 정의
                            param_dist = {
                                'n_estimators': [50, 100, 200, 300, 500],
                                'max_depth': [None, 10, 20, 30, 40, 50],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4]
                            }
                            
                            # 기본 모델
                            base_model = RandomForestRegressor(random_state=42)
                            
                            # RandomizedSearchCV 설정
                            random_search = RandomizedSearchCV(
                                estimator=base_model,
                                param_distributions=param_dist,
                                n_iter=20,  # 시도할 파라미터 조합 수
                                scoring='neg_mean_squared_error',
                                cv=5,  # 5-fold 교차 검증
                                verbose=0,
                                random_state=42,
                                n_jobs=-1  # 모든 CPU 코어 사용
                            )
                            
                            # 하이퍼파라미터 튜닝 실행
                            with st.spinner("하이퍼파라미터 튜닝 중..."):
                                random_search.fit(X_train, y_train)
                            
                            # 최적 파라미터 출력
                            st.success(f"최적 하이퍼파라미터: {random_search.best_params_}")
                            
                            # 최적 모델 선택
                            model = random_search.best_estimator_
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                    elif model_type == "XGBoost":
                        if tune_hyperparams:
                            # 하이퍼파라미터 튜닝
                            from sklearn.model_selection import RandomizedSearchCV
                            
                            # 파라미터 그리드 정의
                            param_dist = {
                                'n_estimators': [50, 100, 200, 300, 500],
                                'max_depth': [3, 5, 7, 9, 11],
                                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                                'subsample': [0.6, 0.8, 1.0],
                                'colsample_bytree': [0.6, 0.8, 1.0]
                            }
                            
                            # 기본 모델
                            base_model = xgb.XGBRegressor(random_state=42)
                            
                            # RandomizedSearchCV 설정
                            random_search = RandomizedSearchCV(
                                estimator=base_model,
                                param_distributions=param_dist,
                                n_iter=20,  # 시도할 파라미터 조합 수
                                scoring='neg_mean_squared_error',
                                cv=5,  # 5-fold 교차 검증
                                verbose=0,
                                random_state=42,
                                n_jobs=-1  # 모든 CPU 코어 사용
                            )
                            
                            # 하이퍼파라미터 튜닝 실행
                            with st.spinner("하이퍼파라미터 튜닝 중..."):
                                random_search.fit(X_train, y_train)
                            
                            # 최적 파라미터 출력
                            st.success(f"최적 하이퍼파라미터: {random_search.best_params_}")
                            
                            # 최적 모델 선택
                            model = random_search.best_estimator_
                        else:
                            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
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
                        with st.expander("💡 회귀 분석 결과 쉽게 이해하기", expanded=False):
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
                            
                            #### 3. MSE와 RMSE는?
                            - **MSE (Mean Squared Error, 평균 제곱 오차)**
                              - 예측값과 실제값의 차이(오차)를 제곱한 것의 평균이에요
                              - **작을수록**: 모델의 예측이 더 정확하다는 의미예요
                              - **클수록**: 모델의 예측이 부정확하다는 의미예요
                              - 단점: 단위가 제곱되어 있어서 직관적이지 않아요
                            
                            - **RMSE (Root Mean Squared Error, 평균 제곱근 오차)**
                              - MSE의 제곱근을 취한 값이에요
                              - **작을수록**: 모델의 예측이 더 정확하다는 의미예요
                              - **클수록**: 모델의 예측이 부정확하다는 의미예요
                              - 장점: 원래 데이터와 같은 단위라서 직관적이에요
                              - 예: RMSE가 5라면 → 예측값이 실제값과 평균적으로 5단위 정도 차이가 난다는 의미예요
                            
                            #### 4. 회귀 방정식 활용법
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
                        
                        try:
                            # 특이 행렬 문제를 방지하기 위해 np.linalg.pinv 사용
                            var_b = mse * np.linalg.pinv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
                            
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
                            with st.expander("💡 회귀 모델 가정 검정 이해하기", expanded=False):
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
                            
                        except np.linalg.LinAlgError:
                            st.error("선형 회귀 분석 중 오류가 발생했습니다. 선택한 변수들 간에 강한 상관관계가 있어 분석이 불가능합니다. 다른 변수 조합을 선택해보세요.")
                            st.info("다중공선성 문제가 발생했을 수 있습니다. 변수들 간의 상관관계를 확인하고 중복된 정보를 제공하는 변수를 제거해보세요.")
                            
                            # 상관관계 히트맵 표시
                            st.subheader("변수 간 상관관계")
                            corr_matrix = X_train.corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                            st.pyplot(fig)
                    
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
                    with st.expander("💡 랜덤 서치(Random Search)란?", expanded=False):
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

                    # 원인 분석 그래프 추가
                    st.write("### 원인 분석")
                    
                    # 설명 부분을 먼저 표시
                    with st.expander("💡 원인 분석 그래프 이해하기", expanded=False):
                        st.markdown("""
                        ### 원인 분석 그래프 쉽게 이해하기
                        
                        이 그래프는 각 변수가 현재 예측값에 미치는 영향을 보여줍니다.
                        마치 요리에서 각 재료가 최종 맛에 미치는 영향력을 보는 것과 같습니다!
                        
                        #### 1. 변수별 기여도 이해하기
                        
                        **기여도란?**
                        - 각 변수가 예측값을 얼마나 변화시키는지를 나타내는 지표입니다
                        - 양수: 예측값을 증가시키는 방향으로 작용 (예: 온도가 높을수록 수율 증가)
                        - 음수: 예측값을 감소시키는 방향으로 작용 (예: 압력이 높을수록 불량률 감소)
                        - 절대값이 클수록: 영향력이 크다는 의미입니다
                        
                        **기여도 해석 예시:**
                        - 기여도 +0.5: 이 변수가 평균보다 높게 설정되어 예측값을 0.5만큼 증가시킴
                        - 기여도 -0.3: 이 변수가 평균보다 낮게 설정되어 예측값을 0.3만큼 감소시킴
                        
                        #### 2. 평균 대비 영향 이해하기
                        
                        **평균 대비 영향이란?**
                        - 각 변수의 현재값이 평균값과 얼마나 차이나는지를 보여줍니다
                        - 양수: 평균보다 높은 값으로 설정됨
                        - 음수: 평균보다 낮은 값으로 설정됨
                        - 백분율(%): 평균 대비 얼마나 차이나는지를 백분율로 표시
                        
                        **차이(%) 해석 예시:**
                        - +20%: 평균보다 20% 높은 값으로 설정됨
                        - -15%: 평균보다 15% 낮은 값으로 설정됨
                        
                        #### 3. 기여도가 낮더라도 중요한 이유
                        
                        **기여도가 낮은 변수도 중요한 경우:**
                        1. **임계값(Threshold) 효과**: 
                           - 특정 값 이하/이상이면 급격한 변화를 일으킬 수 있음
                           - 예: 온도가 80°C 이하면 반응이 일어나지 않지만, 80°C 이상이면 급격히 반응
                        
                        2. **상호작용(Interaction) 효과**: 
                           - 다른 변수와 함께 작용할 때 중요해질 수 있음
                           - 예: 압력과 온도가 모두 높을 때만 특정 효과가 발생
                        
                        3. **안정성(Stability) 요인**: 
                           - 변동이 작더라도 안정적인 공정을 위해 중요할 수 있음
                           - 예: pH 값이 약간만 변해도 품질에 큰 영향을 미칠 수 있음
                        
                        4. **비용 효율성(Cost Efficiency)**: 
                           - 조정 비용이 낮은 변수라면 작은 영향이라도 조정 가치가 있음
                           - 예: 약간의 온도 조정으로 큰 효과를 볼 수 있는 경우
                        
                        #### 4. 원인 분석 활용 방법
                        
                        **최적화 전략:**
                        - **긍정적 영향 변수**: 기여도가 큰 변수는 더 세밀하게 조정
                        - **부정적 영향 변수**: 기여도가 작은 변수는 범위를 넓게 설정
                        - **최적화 방향**: 기여도 방향에 따라 변수 값을 조정
                        
                        **실제 적용 예시:**
                        1. 기여도가 큰 변수(예: +0.8)는 현재 설정값이 적절한지 확인
                        2. 기여도가 음수인 변수(예: -0.5)는 값을 증가시켜 부정적 영향 감소
                        3. 기여도가 낮은 변수도 상호작용 가능성을 고려하여 조정
                        
                        **주의사항:**
                        - 기여도는 현재 설정값 기준으로 계산되므로, 변수 값을 크게 변경하면 기여도도 변할 수 있음
                        - 여러 변수를 동시에 조정할 때는 상호작용 효과를 고려해야 함
                        - 실제 공정에서는 변수 간 제약조건이 있을 수 있으므로 이를 고려해야 함
                        """)
                    
                    # 변수별 기여도 계산
                    contributions = {}
                    mean_values = {}
                    valid_features = []
                    
                    for feature in st.session_state.model_features:
                        try:
                            # 현재 값이 있는지 확인
                            if feature not in st.session_state.last_input_values:
                                st.warning(f"'{feature}' 변수가 입력값에 없어 기여도 계산에서 제외됩니다.")
                                continue
                            
                            current_value = st.session_state.last_input_values[feature]
                            mean_value = numeric_data[feature].mean()
                            mean_values[feature] = mean_value
                            valid_features.append(feature)
                            
                            # 변수의 영향력 계산
                            if st.session_state.model_type == "선형 회귀":
                                # 선형 회귀의 경우 계수를 사용
                                coef = st.session_state.model.coef_[list(st.session_state.model_features).index(feature)]
                                contribution = coef * (current_value - mean_value)
                            else:
                                # RandomForest나 XGBoost의 경우 feature_importances_를 사용
                                importance = st.session_state.model.feature_importances_[list(st.session_state.model_features).index(feature)]
                                contribution = importance * (current_value - mean_value) / mean_value
                            
                            contributions[feature] = contribution
                        except Exception as e:
                            st.warning(f"'{feature}' 변수의 기여도 계산 중 오류 발생: {str(e)}")
                            continue
                    
                    if not contributions:
                        st.error("기여도를 계산할 수 있는 변수가 없습니다.")
                    else:
                        # 기여도를 데이터프레임으로 변환
                        contribution_df = pd.DataFrame({
                            '변수': list(contributions.keys()),
                            '기여도': list(contributions.values()),
                            '평균 대비': [st.session_state.last_input_values[f] - mean_values[f] for f in contributions.keys()]
                        })

                        # 기여도 기준으로 정렬
                        contribution_df = contribution_df.sort_values('기여도', key=abs, ascending=False)

                        # Plotly 그래프 생성
                        fig_contribution = go.Figure()

                        # 기여도 바 차트
                        fig_contribution.add_trace(
                            go.Bar(
                                y=contribution_df['변수'],
                                x=contribution_df['기여도'],
                                orientation='h',
                                marker_color=np.where(contribution_df['기여도'] >= 0, '#3498db', '#e74c3c'),
                                text=[f'{val:.4f}' for val in contribution_df['기여도']],
                                textposition='outside',
                                name='기여도',
                                hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                            )
                        )

                        # 레이아웃 설정
                        fig_contribution.update_layout(
                            title='변수별 기여도 분석',
                            xaxis_title='기여도',
                            yaxis_title='변수',
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=False,
                            yaxis=dict(
                                autorange='reversed'  # 기여도가 큰 순으로 정렬
                            )
                        )

                        # 그래프 표시
                        display_plotly_centered(fig_contribution)

                        # 평균 대비 영향 분석
                        st.write("#### 평균 대비 변수 영향")
                        influence_df = pd.DataFrame({
                            '변수': contribution_df['변수'],
                            '현재값': [st.session_state.last_input_values[f] for f in contribution_df['변수']],
                            '평균값': [mean_values[f] for f in contribution_df['변수']],
                            '차이': [st.session_state.last_input_values[f] - mean_values[f] for f in contribution_df['변수']],
                            '차이(%)': [(st.session_state.last_input_values[f] - mean_values[f]) / mean_values[f] * 100 for f in contribution_df['변수']]
                        })

                        # 스타일 적용
                        st.dataframe(
                            influence_df.style.format({
                                '현재값': '{:.4f}',
                                '평균값': '{:.4f}',
                                '차이': '{:.4f}',
                                '차이(%)': '{:.2f}%'
                            }).background_gradient(cmap='RdYlBu_r', subset=['차이(%)']),
                            use_container_width=True
                        )
            else:
                st.info("먼저 모델을 훈련해주세요.")
    else:
        st.error(f"타깃 변수 '{target_col}'을 찾을 수 없습니다.")
else:
    st.info("시뮬레이션을 시작하려면 CSV 파일을 업로드해주세요.") 


# 페이지 하단 소개

st.markdown("---")
st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ")