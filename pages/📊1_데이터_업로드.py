import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 스테이트 초기화
if 'data' not in st.session_state:
    st.session_state.data = None

# 타이틀 및 설명
st.title('데이터 분석 시스템')
st.markdown('공정 데이터 분석 시스템 프로토타입입니다.')

# 샘플 데이터 생성 함수
def get_sample_data():
    # 더 많은 샘플 데이터 생성 (최소 10개)
    np.random.seed(42)  # 재현성을 위한 시드 설정
    n_samples = 15  # 15개 샘플 생성
    
    # 배치번호 생성
    batch_ids = [f'B{str(i+1).zfill(3)}' for i in range(n_samples)]
    
    # 제품 유형 생성 (3가지)
    products = np.random.choice(['제품A', '제품B', '제품C'], size=n_samples)
    
    # 날짜 생성 (2023-01-01부터 순차적으로)
    import datetime
    start_date = datetime.datetime(2023, 1, 1)
    dates = [(start_date + datetime.timedelta(days=i*5)).strftime('%Y-%m-%d') for i in range(n_samples)]
    
    # 데이터프레임 생성
    data = {
        '배치번호': batch_ids,
        '제품': products,
        '날짜': dates,
        '측정값1': np.random.normal(100, 2, n_samples),  # 평균 100, 표준편차 2인 정규분포
        '측정값2': np.random.normal(65, 3, n_samples),   # 평균 65, 표준편차 3인 정규분포
        '측정값3': np.random.normal(75, 2, n_samples),   # 평균 75, 표준편차 2인 정규분포
        '공정변수1': np.random.normal(15, 0.5, n_samples), # 평균 15, 표준편차 0.5인 정규분포
        '결과1': np.random.normal(85, 3, n_samples),     # 평균 85, 표준편차 3인 정규분포
        '결과2': np.random.normal(95, 2, n_samples)      # 평균 95, 표준편차 2인 정규분포
    }
    
    df = pd.DataFrame(data)
    
    # 소수점 자리 정리
    for col in ['측정값1', '측정값2', '측정값3', '공정변수1', '결과1', '결과2']:
        df[col] = df[col].round(1)
    
    return df

# T-검정 샘플 데이터 생성 함수
def get_ttest_sample_data():
    np.random.seed(42)  # 재현성을 위한 시드 설정
    ttest_data = {
        '라인': ['A'] * 15 + ['B'] * 15, 
        '수율': list(np.random.normal(95, 1.5, 15)) + list(np.random.normal(93, 1.2, 15))
    }
    df = pd.DataFrame(ttest_data)
    df['수율'] = df['수율'].round(2)  # 소수점 자리 정리
    return df

# ANOVA 샘플 데이터 생성 함수
def get_anova_sample_data():
    np.random.seed(42)  # 재현성을 위한 시드 설정
    anova_data = {
        '공급업체': ['X'] * 10 + ['Y'] * 10 + ['Z'] * 10,
        '강도': list(np.random.normal(100, 5, 10)) + list(np.random.normal(105, 4, 10)) + list(np.random.normal(102, 6, 10))
    }
    df = pd.DataFrame(anova_data)
    df['강도'] = df['강도'].round(2)  # 소수점 자리 정리
    return df

# 샘플 데이터 다운로드 링크 생성 함수
def get_sample_download_link():
    df = get_sample_data()
    csv = df.to_csv(index=False, encoding='cp949')
    b64 = base64.b64encode(csv.encode('cp949')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_data.csv">샘플 데이터 다운로드 (.csv)</a>'
    return href

# T-검정 샘플 데이터 다운로드 링크 생성 함수
def get_ttest_sample_download_link():
    df = get_ttest_sample_data()
    csv = df.to_csv(index=False, encoding='cp949')
    b64 = base64.b64encode(csv.encode('cp949')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ttest_sample_data.csv">T-검정 샘플 데이터 다운로드 (.csv)</a>'
    return href

# ANOVA 샘플 데이터 다운로드 링크 생성 함수
def get_anova_sample_download_link():
    df = get_anova_sample_data()
    csv = df.to_csv(index=False, encoding='cp949')
    b64 = base64.b64encode(csv.encode('cp949')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="anova_sample_data.csv">ANOVA 샘플 데이터 다운로드 (.csv)</a>'
    return href

# 데이터 업로드 주의사항 표시
with st.expander("📌 데이터 업로드 주의사항", expanded=True):
    st.markdown("""
    ### 데이터 구조 요구사항
    
    본 시스템은 구조화된 데이터를 필요로 합니다:
    
    1. **파일 형식**: CSV 파일(.csv)로 제공되어야 합니다.
    2. **인코딩**: 한글이 포함된 경우 CP949(EUC-KR) 인코딩을 권장합니다.
    3. **데이터 구조**:
        - 첫 번째 행은 변수명(컬럼명)이어야 합니다.
        - 각 열은 하나의 변수를, 각 행은 하나의 관측값을 나타냅니다.
        - 수치형 데이터는 단위 표시나 콤마가 없어야 합니다(예: '1,000' 대신 '1000').
    4. **제한사항**:
        - 최소 30개 이상의 데이터 포인트(행)가 있을 때 통계적 분석이 유의미합니다.
        - 공정능력분석을 위해서는 수치형 데이터가 필요합니다.
    """)
    
    # 샘플 데이터 표시 및 다운로드 링크
    st.markdown("### 샘플 데이터")
    st.dataframe(get_sample_data(), use_container_width=True)
    st.markdown(get_sample_download_link(), unsafe_allow_html=True)
    
    # T-검정 및 ANOVA 샘플 데이터 표시 및 다운로드 링크
    st.markdown("### 통계분석용 샘플 데이터")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### T-검정 샘플 데이터")
        st.dataframe(get_ttest_sample_data(), use_container_width=True)
        st.markdown(get_ttest_sample_download_link(), unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ANOVA 샘플 데이터")
        st.dataframe(get_anova_sample_data(), use_container_width=True)
        st.markdown(get_anova_sample_download_link(), unsafe_allow_html=True)

# 파일 업로드 함수
def upload_file():
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 파일 인코딩 시도 (여러 인코딩 시도)
            encodings = ['cp949', 'utf-8', 'euc-kr']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"파일이 성공적으로 로드되었습니다. (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                st.error("파일 인코딩을 인식할 수 없습니다. CP949, UTF-8 또는 EUC-KR 인코딩의 CSV 파일을 사용해주세요.")
                return None
            
            st.session_state.data = data
            
            # 데이터 미리보기
            st.subheader('데이터 미리보기')
            st.dataframe(data.head(), use_container_width=True)
            
            # 데이터 통계 요약
            st.subheader('데이터 요약')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("행 수", f"{data.shape[0]:,}개")
            with col2:
                st.metric("열 수", f"{data.shape[1]:,}개")
            with col3:
                st.metric("메모리 사용량", f"{data.memory_usage().sum() / (1024**2):.2f} MB")
            
            # 데이터 유형 확인
            st.subheader("데이터 유형")
            
            # 수치형 및 비수치형 컬럼 분리
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("수치형 변수:", len(numeric_cols))
                st.write(", ".join(numeric_cols))
            with col2:
                st.write("비수치형 변수:", len(non_numeric_cols))
                st.write(", ".join(non_numeric_cols))
            
        except Exception as e:
            st.error(f'파일 처리 중 오류가 발생했습니다: {e}')
            st.session_state.data = None
    
    return st.session_state.data

# 파일 업로드 섹션
data = upload_file()

# 사용 안내
if data is None:
    st.info('상단의 업로드 버튼을 통해 CSV 파일을 업로드하면 분석을 시작할 수 있습니다.')
else:
    st.success('데이터가 준비되었습니다. 상단 메뉴에서 원하는 분석 페이지를 선택하세요!')
    
    # 데이터 검증 정보
    if data.shape[0] < 30:
        st.warning(f"현재 데이터 샘플 수({data.shape[0]}개)가 30개 미만입니다. 통계적으로 유의미한 분석을 위해서는 최소 30개 이상의 샘플이 권장됩니다.")