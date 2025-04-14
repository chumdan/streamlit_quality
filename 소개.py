import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import base64

# 한글 폰트 설정
st.set_page_config(
    page_title="품질예측시스템 소개",
    page_icon="✅",
    layout="wide"
)

# 제목 및 서브 제목
st.title("품질예측시스템 소개")
st.subheader("공정 관리 및 품질 예측 플랫폼")

# 탭 정의
tab1, tab2, tab3 = st.tabs(["시스템 개요", "주요 기능", "개발 일정"])

# 시스템 개요 탭
with tab1:
    st.markdown("## 품질예측시스템이란?")
    
    # 세 개의 컬럼 생성
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 공정 능력 분석")
        st.markdown("""
        - **통계적 품질 관리** 기법 적용
        - 공정 능력 지수(Cp, Cpk) 실시간 분석
        - 관리 한계선 자동 계산 및 모니터링
        - 공정 안정성 분석 및 평가
        """)
        
    with col2:
        st.markdown("### 🤖 머신러닝 기반 예측")
        st.markdown("""
        - 주요 제품별 **품질 예측 모델** 구축
        - 다양한 공정 조건에 따른 시뮬레이션
        - 불량 가능성 사전 예측 및 방지
        - 품질에 영향을 미치는 핵심 요인 분석
        """)
        
    with col3:
        st.markdown("### 📱 데이터 전산화")
        st.markdown("""
        - **IPC(공정 내 품질 검사) 데이터** 통합 관리
        - 히스토리컬 데이터 기반 트렌드 분석
        """)
    
    st.markdown("---")
    
    st.markdown("## 기대 효과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 품질 향상
        - 공정 안정성 향상으로 제품 일관성 확보
        - 핵심 공정 매개변수 최적화
        - 예측 기반 선제적 품질 관리
        """)
        
    with col2:
        st.success("""
        ### 통계적 교육 효과
        - 시스템 사용 과정에서 SPC(통계적 공정관리) 개념과 도구를 자연스럽게 체득
        - 통계 지표 해석 경험을 통한 데이터 기반 품질관리 역량 강화
        - 시각화된 통계 결과물을 통해 직관적인 통계적 사고방식 함양
        - 반복적인 데이터 분석 과정을 통한 통계적 의사결정 능력 향상
        """)

# 주요 기능 탭
with tab2:
    st.markdown("## 주요 기능")
    
    # 메인 기능 4가지 설명
    st.markdown("### 현재 구현된 기능")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1. 공정능력분석
        - 주요 품질 특성에 대한 공정능력지수 계산
        - 정규성 검정 및 통계적 분석
        - 관리도 및 히스토그램 시각화
        - USL/LSL 기준 합격률 분석
        
        #### 2. 상관관계분석
        - 품질 특성 간 상관관계 분석
        - 핵심 인자 간 산점도 분석
        """)
        
    with col2:
        st.markdown("""
        #### 3. 특이값분석
        - 이상 배치에 대한 특이값 분석을 통한 이슈 원인 분석
        - 특이값 발생 패턴 분석
        
        #### 4. 시뮬레이션
        - 머신러닝 기반 품질 예측 모델
        - 공정 조건 변경에 따른 결과 예측
        - 특성 중요도 분석
        - SHAP 기반 변수별 기여도 분석
        """)
    


# 개발 일정 탭
with tab3:
    st.markdown("## 개발 일정")
    
    # 개발 일정 표 생성
    schedule_data = {
        "업무 / 프로젝트": ["AI 품질예측시스템"],
        "1월": [""],
        "2월": [""],
        "3월": ["프로토타입개발"],
        "4월": ["요구사항기획"],
        "5월": [""],
        "6월": [""],
        "7월": ["플랫폼 1차 개발"],
        "8월": [""],
        "9월": ["플랫폼 수정"],
        "10월": [""],
        "11월": ["공정예측기능통합"],
        "12월": [""]
    }
    
    df = pd.DataFrame(schedule_data)
    
    # 스타일링된 테이블 생성
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#0078AA',
            align='center',
            font=dict(color='white', size=14),
            height=40
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[['#F5F5F5']*len(df) if i % 2 == 0 else ['#E1F5FE']*len(df) for i in range(len(df.columns))],
            align=['center'],
            font=dict(size=14),
            height=40
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=120
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 마일스톤 설명
    st.markdown("### 주요 마일스톤")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**2023년 4월**\n\n요구사항 분석 및 기획")
        
    with col2:
        st.info("**2023년 7월**\n\n플랫폼 1차 개발")
        
    with col3:
        st.warning("**2023년 11월**\n\n공정예측기능 플랫폼 통합 및 대상 품목 확대(4개 → 10개)")
    


  
# 페이지 하단 소개
st.markdown("---")
st.markdown("### 품질예측시스템 프로토타입")

st.info("본 프로토타입은 품질예측시스템의 개념과 기능을 설명하기 위해 개발되었습니다.")

st.markdown("**제한사항:**")
st.warning("""
- 이는 정식 프레임워크나 검증된 시스템이 아닌 프로토타입 단계입니다.
- 일부 기능에서 오류가 발생하거나 성능 제한이 있을 수 있습니다.
- 현재 Streamlit 기반으로 구현되어 있어 업무용 프레임워크로서의 안정성과 보안성이 검증되지 않았습니다.
- 대용량 데이터 처리 시 지연이 발생할 수 있습니다.
- 일부 특수한 데이터 형식은 완벽하게 지원되지 않을 수 있습니다.
""")

st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항은 정보기획팀 고동현 주임(내선: 189)에게 피드백 주시면 지속적인 개선에 반영하겠습니다.")