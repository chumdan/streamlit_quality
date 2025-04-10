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
            
            # 모델 선택
            model_type = st.radio(
                "모델 선택:",
                ["RandomForest", "XGBoost"],
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
                    else:  # XGBoost
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                    
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
                    
                    # 모델 및 특성 저장
                    st.session_state.model = model
                    st.session_state.model_features = top_indices.tolist()
                    st.session_state.remove_outliers = remove_outliers
                    st.session_state.apply_scaling = apply_scaling
                    
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
                    if model_type == "RandomForest":
                        # 랜덤 포레스트 변수 중요도
                        feature_importance = model.feature_importances_
                    elif model_type == "XGBoost":
                        # XGBoost 변수 중요도
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
                    
                    st.success("모델 훈련 완료! 이제 '시뮬레이션' 탭으로 이동하여 예측을 수행할 수 있습니다.")
        
        with tab2:
            st.write("### 시뮬레이션")
            
            if 'model' in st.session_state and 'model_features' in st.session_state:
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
                
                # 변수별 기여도 분석 (SHAP 대신 대체 방법 사용)
                if st.checkbox("변수별 기여도 분석 보기", key="feature_impact_checkbox"):
                    with st.spinner("변수 기여도 분석 중..."):
                        try:
                            # 모델 확인
                            if 'model' not in st.session_state:
                                st.error("훈련된 모델이 없습니다. 먼저 '모델 훈련' 탭에서 모델을 훈련해주세요.")
                            else:
                                # 1. 모델 기본 특성 중요도 표시 (가능한 경우)
                                if hasattr(st.session_state.model, 'feature_importances_'):
                                    st.subheader("모델의 전반적인 변수 중요도")
                                    
                                    importances = st.session_state.model.feature_importances_
                                    # 내림차순 정렬하여 상위 10개 선택
                                    indices = np.argsort(importances)[::-1][:10]  # 상위 10개
                                    
                                    feature_importance_df = pd.DataFrame({
                                        '변수': [st.session_state.model_features[i] for i in indices],
                                        '중요도': [importances[i] for i in indices]
                                    })
                                    
                                    # 이미지처럼 수평 막대 그래프 표시 (Plotly 사용)
                                    fig_impact = go.Figure()

                                    # 모든 막대를 남색으로 설정
                                    fig_impact.add_trace(
                                        go.Bar(
                                            y=feature_importance_df['변수'],
                                            x=feature_importance_df['중요도'],
                                            orientation='h',
                                            marker_color='#3498db',  # 모두 남색으로 통일
                                            text=[f'{val:.4f}' for val in feature_importance_df['중요도']],
                                            textposition='outside',
                                            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                                        )
                                    )

                                    # 레이아웃 설정
                                    max_impact = max(importances) if len(importances) > 0 else 0
                                    fig_impact.update_layout(
                                        title=f"모델의 전반적인 변수 중요도",
                                        xaxis_title="중요도",
                                        yaxis=dict(
                                            title="변수",
                                            autorange="reversed"  # 위에서부터 내림차순 정렬
                                        ),
                                        height=500,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        xaxis=dict(
                                            range=[0, max_impact*1.1],  # 0부터 시작하도록 수정
                                            showgrid=True,
                                            gridwidth=1,
                                            gridcolor='LightGrey'
                                        )
                                    )

                                    # 중앙에 표시
                                    display_plotly_centered(fig_impact)
                                
                                # 마지막 예측이 있는 경우에만 변수별 영향도 분석
                                if 'last_prediction' in st.session_state and 'last_input_values' in st.session_state:
                                    # 현재 예측에 대한 변수별 영향도 분석
                                    st.subheader(f"{target_col} 예측에 대한 변수별 영향")
                                    
                                    # 저장된 입력값 사용
                                    input_values = st.session_state.last_input_values
                                    input_df = pd.DataFrame([input_values])
                                    base_prediction = st.session_state.last_prediction
                                    
                                    # 각 변수의 영향도를 개별적으로 테스트
                                    impact_results = []
                                    
                                    # 각 특성에 대해 반복
                                    for feature in st.session_state.model_features:
                                        # 각 특성의 중요도를 측정하기 위해 개별 테스트
                                        feature_min = numeric_data[feature].min()
                                        feature_max = numeric_data[feature].max()
                                        feature_mean = numeric_data[feature].mean()
                                        feature_range = feature_max - feature_min
                                        
                                        # 현재 특성의 값
                                        current_value = input_values[feature]
                                        
                                        # 테스트 케이스 생성 (최소, 평균, 최대값)
                                        test_values = {
                                            '최소값': feature_min,
                                            '평균값': feature_mean,
                                            '최대값': feature_max
                                        }
                                        
                                        # 다양한 값으로 테스트
                                        predictions = {}
                                        for label, value in test_values.items():
                                            # 이미 현재 값과 같으면 건너뛰기
                                            if value == current_value:
                                                predictions[label] = base_prediction
                                                continue
                                                
                                            # 특성 값 변경
                                            modified_input = input_df.copy()
                                            modified_input[feature] = value
                                            
                                            # 변경된 입력으로 예측
                                            if 'apply_scaling' in st.session_state and st.session_state.apply_scaling:
                                                modified_input_scaled = st.session_state.scaler.transform(modified_input)
                                                modified_input_scaled_df = pd.DataFrame(modified_input_scaled, columns=modified_input.columns)
                                                modified_prediction = st.session_state.model.predict(modified_input_scaled_df)[0]
                                            else:
                                                modified_prediction = st.session_state.model.predict(modified_input)[0]
                                            
                                            predictions[label] = modified_prediction
                                        
                                        # 변수의 영향도 계산 (최대-최소 차이)
                                        if len(predictions) > 1:
                                            impact = predictions['최대값'] - predictions['최소값']
                                            # 현재 값이 평균보다 높은지 낮은지에 따라 부호 결정
                                            if current_value > feature_mean:
                                                direction = 1  # 평균보다 높음
                                            else:
                                                direction = -1  # 평균보다 낮음
                                                
                                            # 전체 범위 대비 현재 값의 상대적 위치에 따라 영향도 가중치 부여
                                            relative_position = (current_value - feature_mean) / (feature_range/2) if feature_range > 0 else 0
                                            # 영향도는 변수 범위에서의 변화량 * 현재 값의 상대적 위치로 계산
                                            weighted_impact = impact * relative_position
                                        else:
                                            weighted_impact = 0  # 영향도 측정 불가능한 경우
                                        
                                        # 결과 저장
                                        impact_results.append({
                                            '변수': feature,
                                            '현재값': current_value,
                                            '최소예측': predictions.get('최소값', base_prediction),
                                            '최대예측': predictions.get('최대값', base_prediction),
                                            '영향도': weighted_impact
                                        })
                                    
                                    # 결과를 데이터프레임으로 변환
                                    impact_df = pd.DataFrame(impact_results)
                                    
                                    # 영향도의 절대값 기준으로 정렬
                                    impact_df = impact_df.sort_values(by='영향도', key=abs, ascending=False)
                                    
                                    # 상위 10개 변수만 표시
                                    impact_df = impact_df.head(10)
                                    
                                    # 중복되는 표 제거 (상세 데이터 expander로 충분함)
                                    
                                    # 이미지처럼 수평 막대 그래프 표시 (Plotly 사용)
                                    fig_impact = go.Figure()

                                    # 영향도에 따라 색상 설정
                                    colors = ['#3498db' if x > 0 else '#e74c3c' for x in impact_df['영향도'].values]

                                    # 수평 막대 추가
                                    fig_impact.add_trace(
                                        go.Bar(
                                            y=impact_df['변수'],
                                            x=impact_df['영향도'],
                                            orientation='h',
                                            marker_color=colors,
                                            text=[f'{val:.4f}' for val in impact_df['영향도']],
                                            textposition='outside',
                                            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                                        )
                                    )

                                    # 레이아웃 설정
                                    max_impact = max(abs(np.max(impact_df['영향도'].values)), abs(np.min(impact_df['영향도'].values))) if len(impact_df['영향도'].values) > 0 else 0
                                    fig_impact.update_layout(
                                        title=f"{target_col} 예측에 대한 변수별 영향도",
                                        xaxis_title="예측값에 대한 영향도",
                                        yaxis=dict(
                                            title="변수",
                                            autorange="reversed"  # 위에서부터 내림차순 정렬
                                        ),
                                        height=500,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        xaxis=dict(
                                            range=[-max_impact*1.1, max_impact*1.1] if max_impact > 0 else None,
                                            zeroline=True,
                                            zerolinecolor='gray',
                                            zerolinewidth=1,
                                            showgrid=True,
                                            gridwidth=1,
                                            gridcolor='LightGrey'
                                        )
                                    )

                                    # 중앙에 표시
                                    display_plotly_centered(fig_impact)
                                    
                                    # 상세 데이터 표시
                                    with st.expander("변수별 영향도 상세 데이터"):
                                        st.dataframe(impact_df)
                                    
                                    # 결과 해석 가이드
                                    with st.expander("💡 변수 영향도 해석 방법"):
                                        st.markdown("""
                                        ### 변수 영향도 해석 방법
                                        
                                        이 분석은 각 변수가 현재 예측에 얼마나 영향을 미치는지 보여주는 직관적인 지표입니다.
                                        
                                        #### 쉽게 이해하기
                                        - **빨간색 막대(양수)**: 이 변수는 지금 예측값을 **높이고 있어요**
                                        - **파란색 막대(음수)**: 이 변수는 지금 예측값을 **낮추고 있어요**
                                        - **막대가 길수록**: 변수의 영향력이 크다는 의미입니다
                                        
                                        #### 실제 활용법
                                        - 빨간색(양수) 막대가 큰 변수를 **낮추면** → 예측값이 감소합니다
                                        - 파란색(음수) 막대가 큰 변수를 **낮추면** → 예측값이 증가합니다
                                        - 특정 목표치를 원한다면, 막대가 큰 변수부터 조정하세요
                                        
                                        #### 영향도가 0인 변수는 왜 그럴까요?
                                        영향도가 0으로 나타나는 변수는 다음과 같은 이유가 있습니다:
                                        
                                        1. **현재 상태에서 영향이 미미함**: 다른 변수들이 더 지배적인 영향을 미치고 있습니다
                                        2. **변수의 범위가 좁음**: 최소값과 최대값 사이의 차이가 작아서 변화해도 예측에 큰 영향이 없습니다
                                        3. **모델의 특성**: 모델이 이 변수에 대해 학습한 영향력이 작거나, 다른 변수와의 상호작용에서만 중요합니다
                                        4. **비선형 관계**: 현재 값을 중심으로는 영향이 적지만, 다른 구간에서는 영향이 클 수 있습니다
                                        
                                        > 💡 **중요**: 영향도가 0이라도 반드시 중요하지 않은 것은 아닙니다! 다른 상황이나 다른 변수값과 조합될 때 중요해질 수 있습니다.
                                        """)
                                else:
                                    st.info("예측을 먼저 수행해주세요.")
                                
                        except Exception as e:
                            import traceback
                            st.error(f"변수 기여도 분석 중 오류가 발생했습니다: {e}")
                            st.code(traceback.format_exc(), language="python")
            else:
                st.info("먼저 모델을 훈련해주세요.")
    else:
        st.error(f"타깃 변수 '{target_col}'을 찾을 수 없습니다.")
else:
    st.info("시뮬레이션을 시작하려면 CSV 파일을 업로드해주세요.") 