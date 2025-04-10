import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 페이지 설정
st.set_page_config(
    page_title="제약 공정 데이터 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 스테이트 초기화
if 'data' not in st.session_state:
    st.session_state.data = None

# 타이틀 및 설명
st.title('제약 공정 데이터 분석 시스템')
st.markdown('프리그렐정 데이터 기반 공정 분석 시스템 프로토타입입니다.')

# 파일 업로드 함수
def upload_file():
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 파일 인코딩 시도 (여러 인코딩 시도)
            encodings = ['cp949', 'utf-8', 'euc-kr']
            for encoding in encodings:
                try:
                    data = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            # 컬럼명 정리
            rename_dict = {
                '제조번호': '제조번호', '수분함량(%)': '수분함량(%)', 'd(0.1)': 'd(0.1)', 
                'd(0.5)': 'd(0.5)', 'd(0.9)': 'd(0.9)', '제품명': '제품명', 
                '타정기': '타정기', '타정 일자': '타정 일자', '기표타정중량(mg)': '기표타정중량(mg)', 
                '평균질량(mg)': '평균질량(mg)', '질량범위_Min(mg)': '질량범위_Min(mg)', 
                '질량범위_Max(mg)': '질량범위_Max(mg)', '경도_Min(N)': '경도_Min(N)', 
                '경도_Max(N)': '경도_Max(N)', '두께_Min': '두께_Min', '두께_Max': '두께_Max', 
                '마손도(%)': '마손도(%)', '붕해도(min)': '붕해도(min)', '충전깊이(mm)': '충전깊이(mm)', 
                '호퍼오리피스(mm)': '호퍼오리피스(mm)', '다이직경(mm)': '다이직경(mm)', 
                '타정속도(rpm)': '타정속도(rpm)', '타압력(KN)': '타압력(KN)', 
                '혼합기 속도(rpm)': '혼합기 속도(rpm)', '용출_최소': '용출_최소', 
                '용출_최대': '용출_최대', '작업자': '작업자'
            }
            
            # 데이터프레임의 열 이름을 재정의
            data.rename(columns=lambda x: rename_dict.get(x, x), inplace=True)
            
            st.session_state.data = data
            st.success('파일이 성공적으로 업로드되었습니다!')
            
            # 데이터 미리보기
            st.subheader('데이터 미리보기')
            st.dataframe(data.head())
            
            # 데이터 통계 요약
            st.subheader('데이터 요약')
            st.write(f"행 수: {data.shape[0]}, 열 수: {data.shape[1]}")
            
            # 데이터 유형 확인
            st.write("데이터 유형:")
            st.write(data.dtypes)
            
        except Exception as e:
            st.error(f'파일 처리 중 오류가 발생했습니다: {e}')
            st.session_state.data = None
    
    return st.session_state.data

# 파일 업로드 섹션
data = upload_file()

# 사용 안내
if data is None:
    st.info('왼쪽 메뉴에서 CSV 파일을 업로드한 후, 상단의 페이지 탭을 통해 다양한 분석을 수행할 수 있습니다.')
    
    # 샘플 이미지 표시
    st.subheader('시스템 기능 설명')
    st.markdown("""
    ### 주요 기능:
    
    1. **공정 능력분석**:
       - 변수별 통계량 및 공정능력지수(Cp, Cpk) 계산
       - 공정 관리도 및 분포 시각화
       - 공정 능력 판정 및 해석
    
    2. **통계분석**:
       - 변수 간 상관관계 분석 및 시각화
       - 주요 상관변수와의 산점도 및 회귀선
       - 선택 변수의 박스플롯 및 기술통계량
    
    3. **특이값분석**:
       - 배치별 Z-Score 계산 및 시각화
       - 특이값 해석 및 상세 분석
       - 이상값 분포 분석
    4. **시뮬레이션**:
       - 업로드한 데이터 학습 및 시뮬레이션 결과 출력

    """)
else:
    st.success('데이터가 준비되었습니다. 상단 메뉴에서 원하는 분석 페이지를 선택하세요!')