import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import io
import base64
from matplotlib.figure import Figure
import plotly.express as px
import plotly.figure_factory as ff
from plotly.io import to_image
import plotly.io as pio
import warnings

# kaleido 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 보고서 생성 함수 정의
def create_csv_data():
    """공정능력분석 결과를 CSV 형식으로 생성"""
    # 현재 스코프에서 전역 변수 사용
    global data, selected_var, var_data, var_data_original, mean_val, std_val, min_val, max_val, lsl, usl
    global cp, cpk, cpu, cpl, yield_rate, defect_rate_ppm, normality_result
    
    # CSV용 데이터 프레임 생성
    result_df = pd.DataFrame()
    
    # 기본 정보 섹션
    info_data = {
        "항목": ["분석 변수", "데이터 개수", "평균", "표준편차", "최소값", "최대값", "하한규격(LSL)", "상한규격(USL)", "정규성 검정 결과"],
        "값": [selected_var, len(var_data), mean_val, std_val, min_val, max_val, lsl, usl, normality_result]
    }
    info_df = pd.DataFrame(info_data)
    
    # 공정능력 지수 섹션
    capability_data = {
        "항목": ["Cp/Pp", "Cpk/Ppk", "Cpu/Ppu", "Cpl/Ppl", "규격 내 비율(%)", "불량률(PPM)"],
        "값": [cp, cpk, cpu, cpl, yield_rate, defect_rate_ppm]
    }
    capability_df = pd.DataFrame(capability_data)
    
    # 원본 데이터 준비
    data_df = pd.DataFrame({selected_var: var_data_original})
    
    # CSV 데이터 생성 (BOM 추가로 한글 문제 해결)
    buffer = io.StringIO()
    buffer.write('\ufeff')  # BOM 문자 추가
    
    # 섹션 구분자와 함께 각 데이터프레임 기록
    buffer.write("# 기본 정보\n")
    info_df.to_csv(buffer, index=False, encoding='utf-8')
    
    buffer.write("\n\n# 공정능력 지수\n")
    capability_df.to_csv(buffer, index=False, encoding='utf-8')
    
    buffer.write("\n\n# 원본 데이터\n")
    data_df.to_csv(buffer, index=True, encoding='utf-8')
    
    return buffer.getvalue()

def create_html_report():
    """공정능력분석 결과를 HTML 보고서 형식으로 생성 (그래프 제외)"""
    # st.write("--- DEBUG: Inside create_html_report (No Graphs): Starting --- ") # 디버깅 제거
    # 현재 스코프에서 전역 변수 사용
    global data, selected_var, var_data, var_data_original, mean_val, std_val, min_val, max_val, lsl, usl
    global cp, cpk, cpu, cpl, yield_rate, defect_rate_ppm, normality_result, shapiro_result
    # 그래프 관련 global 변수 제거

    # 이미지 변환 함수 제거 (이미 없음)

    # 전체 HTML 생성 과정을 try-except로 감쌈
    try:
        # st.write("--- DEBUG: Inside create_html_report (No Graphs): Preparing data --- ") # 디버깅 제거
        
        # --- 그래프 변환 관련 로직 완전 제거 ---
        # === 임시 테스트 코드 완전 제거 ===

        # CSS 클래스 결정 함수들 (이전과 동일, 내용 축약)
        def get_cp_class():
            if cp >= 1.33: return 'good'
            elif cp >= 1.0: return 'warning'
            else: return 'bad'
        def get_cpk_class():
            if cpk >= 1.33: return 'good'
            elif cpk >= 1.0: return 'warning'
            else: return 'bad'
        def get_yield_class():
            if yield_rate >= 99.73: return 'good'
            elif yield_rate >= 95: return 'warning'
            else: return 'bad'
        def get_defect_class():
            if defect_rate_ppm <= 2700: return 'good'
            elif defect_rate_ppm <= 50000: return 'warning'
            else: return 'bad'

        # 결과 텍스트 함수들 (원래 로직 복원)
        def get_cp_text():
            if cp >= 1.33: return '우수 (Cp ≥ 1.33)'
            elif cp >= 1.0: return '적절 (1.00 ≤ Cp < 1.33)'
            else: return '부적합 (Cp < 1.00)'
        def get_cpk_text():
            if cpk >= 1.33: return '우수 (Cpk ≥ 1.33)'
            elif cpk >= 1.0: return '적절 (1.00 ≤ Cpk < 1.33)'
            else: return '부적합 (Cpk < 1.00)'
        def get_yield_text():
            if yield_rate >= 99.73: return '양호 (≥ 99.73%)'
            elif yield_rate >= 95: return '주의 (≥ 95%)'
            else: return '개선필요 (< 95%)'
        def get_defect_text():
            if defect_rate_ppm <= 2700: return '양호 (≤ 2,700 PPM)'
            elif defect_rate_ppm <= 50000: return '주의 (≤ 50,000 PPM)'
            else: return '개선필요 (> 50,000 PPM)'

        # 결과 해석 텍스트 (원래 로직 복원)
        def get_capability_text():
            if cpk >= 1.33:
                return f'공정이 규격 요구사항을 <span class="good">충분히 만족</span>합니다. (Cpk = {cpk:.2f} ≥ 1.33)'
            elif cpk >= 1.0:
                return f'공정이 규격 요구사항을 <span class="warning">최소한으로 만족</span>합니다. (Cpk = {cpk:.2f})'
            else:
                return f'공정이 규격 요구사항을 <span class="bad">만족하지 못합니다</span>. (Cpk = {cpk:.2f} < 1.0)'
        def get_center_text():
            if not np.isnan(lsl) and not np.isnan(usl) and not np.isnan(std_val) and std_val > 0:
                spec_center = (lsl + usl) / 2
                deviation = abs(mean_val - spec_center)
                if deviation < 0.1 * std_val:
                    return f'공정 평균({mean_val:.2f})이 규격 중심({spec_center:.2f})에 <span class="good">매우 가깝습니다</span>.'
                elif deviation < 0.5 * std_val:
                    return f'공정 평균({mean_val:.2f})이 규격 중심({spec_center:.2f})과 <span class="warning">약간 차이</span>가 있습니다.'
                else:
                    return f'공정 평균({mean_val:.2f})이 규격 중심({spec_center:.2f})과 <span class="bad">상당한 차이</span>가 있습니다.'
            else:
                return f'공정 평균({mean_val:.2f}) (규격 중심과의 비교 불가)'
        def get_dispersion_text():
            if cp >= 1.33:
                return f'공정 산포가 <span class="good">충분히 작습니다</span>. (Cp = {cp:.2f} ≥ 1.33)'
            elif cp >= 1.0:
                return f'공정 산포가 <span class="warning">경계 수준</span>입니다. (Cp = {cp:.2f})'
            else:
                return f'공정 산포가 <span class="bad">너무 큽니다</span>. (Cp = {cp:.2f} < 1.0)'
        def get_improvement_text():
            recommendations = []
            if cpk < 1.33:
                # 중심 개선 제안
                if not np.isnan(lsl) and not np.isnan(usl) and not np.isnan(std_val) and std_val > 0 and abs(mean_val - (lsl+usl)/2) >= 0.1*std_val:
                    recommendations.append(f'<li>공정 평균을 규격 중심({(lsl+usl)/2:.2f})에 더 가깝게 조정하세요.</li>')
                # 산포 개선 제안
                if cp < 1.33:
                    recommendations.append('<li>공정 변동성을 줄이기 위한 방안을 검토하세요 (원인 분석, 표준화 강화 등).</li>')
                recommendations.append('<li>공정 관리 시스템을 강화하고 정기적인 모니터링을 실시하세요.</li>')
            else:
                recommendations.append('<li>현재 공정이 규격을 충분히 만족하므로 현 상태 유지 및 관리에 집중하세요.</li>')
                
            return "".join(recommendations)

        # 그래프 섹션 HTML 제거 (이미 없음)

        # HTML 구조 생성 (그래프 섹션 제거 및 섹션 번호 조정)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{selected_var} 공정능력분석 보고서</title>
            <style>
                /* ... (스타일 정의, 그래프 관련 스타일 제거) ... */
                body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
                .container {{ margin-bottom: 30px; }}
                .note {{ background-color: #f8f9fa; padding: 10px; border-left: 5px solid #4CAF50; margin-bottom: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 10px; text-align: center; margin-top: 30px; }}
                .warning-container {{ background-color: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; margin: 20px 0; }}
                @media print {{
                    .header {{ background-color: #fff; color: #000; }}
                    .note {{ background-color: #fff; border-left: 2px solid #000; }}
                    .footer {{ background-color: #fff; }}
                    .warning-container {{ background-color: #fff; border-left: 2px solid #000; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{selected_var} 공정능력분석 보고서</h1>
                <p>생성일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="container">
                <h2>1. 기본 정보</h2>
                <table>
                    <tr><th>항목</th><th>값</th></tr>
                    <tr><td>분석 변수</td><td>{selected_var}</td></tr>
                    <tr><td>데이터 개수</td><td>{len(var_data)}</td></tr>
                    <tr><td>평균</td><td>{mean_val:.4f}</td></tr>
                    <tr><td>표준편차</td><td>{std_val:.4f}</td></tr>
                    <tr><td>최소값</td><td>{min_val:.4f}</td></tr>
                    <tr><td>최대값</td><td>{max_val:.4f}</td></tr>
                    <tr><td>하한규격(LSL)</td><td>{lsl:.4f}</td></tr>
                    <tr><td>상한규격(USL)</td><td>{usl:.4f}</td></tr>
                    <tr><td>정규성 검정 결과</td><td>{normality_result}</td></tr>
                    <tr><td>Shapiro-Wilk 검정</td><td>{shapiro_result if shapiro_result else "N/A"}</td></tr>
                </table>
            </div>
            
            <div class="container">
                <h2>2. 공정능력 분석 결과</h2>
                <table>
                    <tr><th>항목</th><th>값</th><th>평가</th></tr>
                    <tr><td>공정능력지수(Cp/Pp)</td><td>{cp:.4f}</td><td class="{get_cp_class()}">{get_cp_text()}</td></tr>
                    <tr><td>공정능력지수K(Cpk/Ppk)</td><td>{cpk:.4f}</td><td class="{get_cpk_class()}">{get_cpk_text()}</td></tr>
                    <tr><td>상한 공정능력지수(Cpu/Ppu)</td><td>{cpu:.4f}</td><td></td></tr>
                    <tr><td>하한 공정능력지수(Cpl/Ppl)</td><td>{cpl:.4f}</td><td></td></tr>
                    <tr><td>합격률(%)</td><td>{yield_rate:.4f}%</td><td class="{get_yield_class()}">{get_yield_text()}</td></tr>
                    <tr><td>불량률(PPM)</td><td>{defect_rate_ppm:.1f} PPM</td><td class="{get_defect_class()}">{get_defect_text()}</td></tr>
                </table>
            </div>
            
            <div class="container">
                <h2>3. 분석 결과 해석</h2>
                <div class="note">
                    <h3>공정능력 평가</h3><p>{get_capability_text()}</p>
                    <h3>공정 중심 평가</h3><p>{get_center_text()}</p>
                    <h3>공정 산포 평가</h3><p>{get_dispersion_text()}</p>
                </div>
            </div>
            
            <!-- 시각화 자료 섹션 완전 제거 -->
            
            <div class="container">
                <h2>4. 개선 권장사항</h2> <!-- 섹션 번호 수정 -->
                <ul>
                    {get_improvement_text()}
                </ul>
            </div>
            
            <div class="footer">
                <p>이 보고서는 자동으로 생성되었습니다. © 품질관리시스템</p>
            </div>
        </body>
        </html>
        """
        # st.write("--- DEBUG: Inside create_html_report (No Graphs): HTML structure created successfully --- ") # 디버깅 제거
        return html

    except Exception as e:
        st.error(f"HTML 보고서 생성 중 심각한 오류가 발생했습니다: {str(e)}")
        # st.write(f"--- DEBUG: Inside create_html_report (No Graphs): Major error occurred: {e} --- ") # 디버깅 제거
        # 간단한 오류 보고서 반환 (그래프 제외 버전)
        return f"""
        <!DOCTYPE html><html><head><meta charset="UTF-8"><title>오류 보고서</title><style>body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 20px; }}.error {{ color: red; background-color: #ffeeee; padding: 20px; border-left: 5px solid red; }}</style></head><body><h1>보고서 생성 오류</h1><div class="error"><p>HTML 보고서를 생성하는 중 오류가 발생했습니다:</p><p>{str(e)}</p><p>관리자에게 문의하거나 나중에 다시 시도해주세요.</p></div></body></html>
        """

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
    
    ### 관리한계선과 시그마(σ) 레벨
    
    관리도에서 시그마(σ) 레벨은 공정의 변동성을 모니터링하는 기준이 됩니다.
    """)
    
    # 이미지 별도 표시
    st.image("./image/normal distribution sigma levels.png", 
            caption="시그마(σ) 레벨과 데이터 포함 범위",
            width=600)
    
    st.markdown("""
    - **±1σ**: 데이터의 68.27% 포함 (너무 민감)
    - **±2σ**: 데이터의 95.45% 포함 (다소 민감)
    - **±3σ**: 데이터의 99.73% 포함 (적절한 균형)
    
    ±3σ를 기본 관리한계선으로 사용하는 이유:
    1. 통계적 의미: 정규분포에서 데이터의 99.73%를 포함
    2. 실용성: 자연적 변동과 특수원인을 효과적으로 구분
    3. 산업 표준: 월터 슈하트가 제안한 이후 글로벌 표준으로 정착
    4. 균형: 불필요한 경보(false alarm)와 문제 감지 사이의 최적 지점

    ### 주요 지표 (정규분포 가정 시)
    
    - **Cp (공정능력지수)**: 공정의 산포와 규격 폭의 비율
      - Cp = (USL - LSL) / (6σ)
      - **해석 기준**:
        - Cp ≥ 1.33: 우수 (공정이 매우 안정적)
        - 1.00 ≤ Cp < 1.33: 적절 (공정이 관리 가능한 수준)
        - Cp < 1.00: 부적합 (공정 개선 필요)
    
    - **Cpk (공정능력지수K)**: 공정의 산포와 중심이탈을 함께 고려
      - Cpk = min[(USL - μ) / (3σ), (μ - LSL) / (3σ)]
      - **해석 기준**:
        - Cpk ≥ 1.33: 우수 (공정이 규격 중심에 잘 맞춰져 있음)
        - 1.00 ≤ Cpk < 1.33: 적절 (공정이 규격을 만족하나 개선 여지 있음)
        - Cpk < 1.00: 부적합 (공정이 규격을 벗어날 위험이 높음)
    
    - **Cpu (상한 공정능력지수)**: 상한규격에 대한 공정능력
      - Cpu = (USL - μ) / (3σ)
    
    - **Cpl (하한 공정능력지수)**: 하한규격에 대한 공정능력
      - Cpl = (μ - LSL) / (3σ)
      
    ### 비모수적 지표 (정규분포 가정이 성립하지 않을 때)
    
    - **Pp (백분위수 기반 공정능력지수)**: 
      - Pp = (USL - LSL) / (P99.865 - P0.135)
      - 여기서 P99.865와 P0.135는 각각 99.865% 및 0.135% 백분위수
      - **해석 기준**:
        - Pp ≥ 1.33: 우수 (공정이 매우 안정적)
        - 1.00 ≤ Pp < 1.33: 적절 (공정이 관리 가능한 수준)
        - Pp < 1.00: 부적합 (공정 개선 필요)
    
    - **Ppk (백분위수 기반 공정능력지수K)**: 
      - Ppk = min[(USL - P50) / (P99.865 - P50), (P50 - LSL) / (P50 - P0.135)]
      - 여기서 P50은 중앙값(50% 백분위수)
      - **해석 기준**:
        - Ppk ≥ 1.33: 우수 (공정이 규격 중심에 잘 맞춰져 있음)
        - 1.00 ≤ Ppk < 1.33: 적절 (공정이 규격을 만족하나 개선 여지 있음)
        - Ppk < 1.00: 부적합 (공정이 규격을 벗어날 위험이 높음)
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
                    # 이상치 데이터만 필터링하여 표시
                    outlier_data = pd.DataFrame({
                        selected_var: var_data_original[outliers],
                        '원본 인덱스': var_data_original[outliers].index
                    }).reset_index(drop=True)
                    st.dataframe(outlier_data)
                    
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
                    
                    # 규격 내 제품 비율(합격률) 계산
                    z_usl = (usl - mean_val) / std_val
                    z_lsl = (lsl - mean_val) / std_val
                    
                    # 정규분포 가정 하에 합격률 계산
                    prob_above_lsl = stats.norm.cdf(z_lsl)
                    prob_below_usl = stats.norm.cdf(z_usl)
                    
                    # 규격 내 비율(%)
                    yield_rate = (prob_below_usl - prob_above_lsl) * 100
                    # 불량률(PPM)
                    defect_rate_ppm = (1 - (prob_below_usl - prob_above_lsl)) * 1000000
                    
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
                    
                    # 비모수적 방법으로 합격률 계산 (실제 데이터 분포 사용)
                    within_spec = ((var_data >= lsl) & (var_data <= usl)).sum()
                    total_count = len(var_data)
                    
                    # 규격 내 비율(%)
                    yield_rate = (within_spec / total_count) * 100
                    # 불량률(PPM)
                    defect_rate_ppm = ((total_count - within_spec) / total_count) * 1000000
                    
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
                yield_rate = np.nan
                defect_rate_ppm = np.nan
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
            
            # 정규성 시각화(QQ-Plot) 부분을 Plotly로 변경
            # 새로운 Plotly QQ-Plot 코드로 변경
            st.subheader("정규성 시각화 (QQ Plot)")
            st.caption("QQ Plot은 데이터가 정규분포를 따르는지 시각적으로 확인할 수 있는 도구입니다. 직선에 가까울수록 정규분포에 가깝습니다.")

            # QQ 플롯 데이터 생성
            qq_data = stats.probplot(var_data, dist="norm", fit=True)
            theoretical_quantiles = qq_data[0][0]
            sample_quantiles = qq_data[0][1]
            slope, intercept, r = qq_data[1]

            # Plotly QQ Plot 생성
            fig_qq = go.Figure()

            # 데이터 포인트 추가
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles, 
                y=sample_quantiles,
                mode='markers',
                name='데이터',
                marker=dict(color='blue', size=8),
                hovertemplate='이론적 분위수: %{x:.2f}<br>실제 분위수: %{y:.2f}<extra></extra>'
            ))

            # 참조선(직선) 추가
            line_x = np.linspace(min(theoretical_quantiles), max(theoretical_quantiles), 100)
            line_y = slope * line_x + intercept
            fig_qq.add_trace(go.Scatter(
                x=line_x, 
                y=line_y,
                mode='lines',
                name='참조선',
                line=dict(color='red', width=2, dash='solid'),
                hovertemplate='이론적 분위수: %{x:.2f}<br>예상 분위수: %{y:.2f}<extra></extra>'
            ))

            # 레이아웃 설정
            fig_qq.update_layout(
                title=f"Normal Q-Q Plot (R² = {r**2:.4f})",
                xaxis_title='이론적 분위수 (Theoretical Quantiles)',
                yaxis_title='실제 분위수 (Sample Quantiles)',
                hovermode='closest',
                height=500,
                margin=dict(t=50, b=50, l=50, r=50),
                showlegend=True
            )

            # 그리드 추가
            fig_qq.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig_qq.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

            # Plotly 그래프 표시
            display_plotly_centered(fig_qq)

            # QQ Plot 해석
            if normality_result == "정규 분포 (p >= 0.05)":
                r_squared = r**2
                if r_squared > 0.95:
                    st.success(f"✅ QQ Plot 해석: 데이터가 정규분포를 매우 잘 따릅니다. (R² = {r_squared:.4f})")
                else:
                    st.success(f"✅ QQ Plot 해석: 데이터가 대체로 정규분포를 따릅니다. (R² = {r_squared:.4f})")
            else:
                r_squared = r**2
                if r_squared < 0.90:
                    st.warning(f"⚠️ QQ Plot 해석: 데이터가 정규분포를 따르지 않습니다. (R² = {r_squared:.4f})")
                else:
                    st.warning(f"⚠️ QQ Plot 해석: 데이터가 정규분포와 약간 차이가 있습니다. (R² = {r_squared:.4f})")
            st.info("""
            💡 **R² 값이란?**
            - R²(결정계수)는 QQ Plot에서 점들이 직선에 얼마나 잘 맞는지를 나타내는 지표입니다
            - 0부터 1 사이의 값을 가지며, 1에 가까울수록 정규분포에 가깝습니다
            - 일반적으로 R² > 0.95면 매우 좋음, R² > 0.90이면 양호한 수준으로 판단합니다
            """)

            st.info("""
            💡 **R² 값과 정규성 검정 결과가 다른 이유**
            1. **검정 방법의 차이**:
               - 정규성 검정: 엄격한 통계적 검정으로, 작은 차이도 민감하게 감지
               - QQ Plot의 R²: 시각적/실용적 관점에서 전반적인 정규성 정도를 평가
            
            2. **해석의 차이**:
               - 정규성검정 p < 0.05: 통계적으로 정규분포가 아님을 의미
               - R² > 0.90: 실용적 관점에서 정규분포에 근사함을 의미
            
            3. **활용**:
               - 엄격한 통계적 가정이 필요한 경우: 정규성 검정 결과 사용
               - 실무적 판단이 필요한 경우: R² 값도 함께 고려
            """)



            # 합격률 및 공정능력 지수 표시
            st.subheader("합격률 및 공정능력 지수")

            # 합격률과 불량률 표시 - 3개 컬럼으로 분할
            metrics_row1_col1, metrics_row1_col2, metrics_row1_col3 = st.columns(3)

            with metrics_row1_col1:
                if not np.isnan(yield_rate):
                    st.metric("합격률", f"{yield_rate:.2f}%", 
                            delta="양호" if yield_rate >= 99.73 else 
                                 "주의" if yield_rate >= 95 else 
                                 "개선필요")
                    st.caption("규격 내 제품 비율")
                else:
                    st.metric("합격률", "N/A")
                    st.caption("계산 불가")

            with metrics_row1_col2:
                if not np.isnan(defect_rate_ppm):
                    st.metric("불량률", f"{defect_rate_ppm:.0f} PPM", 
                            delta="양호" if defect_rate_ppm <= 2700 else 
                                 "주의" if defect_rate_ppm <= 50000 else 
                                 "개선필요",
                            delta_color="inverse",
                            help="PPM(Parts Per Million): 백만 개당 불량품의 개수")
                    st.caption("백만 개당 불량 개수")
                else:
                    st.metric("불량률", "N/A",
                            help="PPM(Parts Per Million): 백만 개당 불량품의 개수")
                    st.caption("계산 불가")

            with metrics_row1_col3:
                st.metric("분석 방법", method_used)
                st.caption("데이터 특성에 따른 방법")

            # 기존 공정능력지수 표시
            metrics_row2_col1, metrics_row2_col2, metrics_row2_col3, metrics_row2_col4 = st.columns(4)

            with metrics_row2_col1:
                # 공정능력지수 표시
                cp_display = f"{cp:.2f}" if not np.isnan(cp) else "N/A"
                cp_name = "Cp" if normality_result == "정규 분포 (p >= 0.05)" else "Pp"
                st.metric(cp_name, cp_display, 
                         delta="주의 필요" if not np.isnan(cp) and cp >= 1 and cp < 1.33 else
                               "적합" if not np.isnan(cp) and cp >= 1.33 else
                               "부적합" if not np.isnan(cp) and cp < 1 else "계산 불가")
                st.caption("공정의 산포가 규격 대비 얼마나 좁은지")

            with metrics_row2_col2:
                cpk_display = f"{cpk:.2f}" if not np.isnan(cpk) else "N/A"
                cpk_name = "Cpk" if normality_result == "정규 분포 (p >= 0.05)" else "Ppk"
                st.metric(cpk_name, cpk_display, 
                         delta="주의 필요" if not np.isnan(cpk) and cpk >= 1 and cpk < 1.33 else
                               "적합" if not np.isnan(cpk) and cpk >= 1.33 else
                               "부적합" if not np.isnan(cpk) and cpk < 1 else "계산 불가")
                st.caption("공정 산포와 중심위치를 모두 고려한 지수")

            with metrics_row2_col3:
                cpu_display = f"{cpu:.2f}" if not np.isnan(cpu) else "N/A"
                cpu_name = "Cpu" if normality_result == "정규 분포 (p >= 0.05)" else "Ppu"
                st.metric(cpu_name, cpu_display)
                st.caption("상한규격 기준 공정능력")

            with metrics_row2_col4:
                cpl_display = f"{cpl:.2f}" if not np.isnan(cpl) else "N/A"
                cpl_name = "Cpl" if normality_result == "정규 분포 (p >= 0.05)" else "Ppl"
                st.metric(cpl_name, cpl_display)
                st.caption("하한규격 기준 공정능력")

            # 분포 및 합격률 시각화를 Plotly로 변경
            # 히스토그램과 분포 시각화 - 합격률 시각적 표현
            st.subheader("분포 및 합격률 시각화")

            # 히스토그램 데이터 준비
            hist_values, hist_bins = np.histogram(var_data, bins=20, density=True)
            bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            bin_width = hist_bins[1] - hist_bins[0]

            # Plotly 분포 시각화
            fig_hist = go.Figure()

            # 히스토그램 추가
            fig_hist.add_trace(go.Bar(
                x=bin_centers,
                y=hist_values,
                width=bin_width * 0.9,
                name='관측 데이터',
                marker_color='skyblue',
                hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
            ))

            # 범위 설정
            x_range = np.linspace(min_val - 0.5*std_val, max_val + 0.5*std_val, 200)

            # 밀도 곡선 추가 (정규분포 또는 KDE)
            if normality_result == "정규 분포 (p >= 0.05)":
                # 정규분포 곡선
                y_norm = stats.norm.pdf(x_range, mean_val, std_val)
                fig_hist.add_trace(go.Scatter(
                    x=x_range,
                    y=y_norm,
                    mode='lines',
                    name='정규분포 곡선',
                    line=dict(color='blue', width=2),
                    hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
                ))
            else:
                # KDE 곡선
                kde = gaussian_kde(var_data)
                y_kde = kde(x_range)
                fig_hist.add_trace(go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode='lines',
                    name='KDE 곡선',
                    line=dict(color='blue', width=2),
                    hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
                ))

            # 규격 이탈 영역 (LSL 미만) 추가
            x_lsl = np.linspace(min_val - 0.5*std_val, lsl, 50)
            if normality_result == "정규 분포 (p >= 0.05)":
                y_lsl = stats.norm.pdf(x_lsl, mean_val, std_val)
            else:
                kde = gaussian_kde(var_data)
                y_lsl = kde(x_lsl)

            fig_hist.add_trace(go.Scatter(
                x=x_lsl,
                y=y_lsl,
                mode='none',
                name='하한 규격 이탈',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                hoverinfo='skip'
            ))

            # 규격 이탈 영역 (USL 초과) 추가
            x_usl = np.linspace(usl, max_val + 0.5*std_val, 50)
            if normality_result == "정규 분포 (p >= 0.05)":
                y_usl = stats.norm.pdf(x_usl, mean_val, std_val)
            else:
                kde = gaussian_kde(var_data)
                y_usl = kde(x_usl)

            fig_hist.add_trace(go.Scatter(
                x=x_usl,
                y=y_usl,
                mode='none',
                name='상한 규격 이탈',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                hoverinfo='skip'
            ))

            # 수직선 추가
            # LSL, USL 수직선
            fig_hist.add_trace(go.Scatter(
                x=[lsl, lsl],
                y=[0, max(hist_values)*1.2],
                mode='lines',
                name='하한규격(LSL)',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate=f'LSL: {lsl:.2f}<extra></extra>'
            ))

            fig_hist.add_trace(go.Scatter(
                x=[usl, usl],
                y=[0, max(hist_values)*1.2],
                mode='lines',
                name='상한규격(USL)',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate=f'USL: {usl:.2f}<extra></extra>'
            ))

            # 평균선
            fig_hist.add_trace(go.Scatter(
                x=[mean_val, mean_val],
                y=[0, max(hist_values)*1.2],
                mode='lines',
                name='평균',
                line=dict(color='green', width=2),
                hovertemplate=f'평균: {mean_val:.2f}<extra></extra>'
            ))

            # +/-3σ 선
            fig_hist.add_trace(go.Scatter(
                x=[mean_val + 3*std_val, mean_val + 3*std_val],
                y=[0, max(hist_values)*1.2],
                mode='lines',
                name='+3σ',
                line=dict(color='orange', width=1.5, dash='dot'),
                hovertemplate=f'+3σ: {mean_val + 3*std_val:.2f}<extra></extra>'
            ))

            fig_hist.add_trace(go.Scatter(
                x=[mean_val - 3*std_val, mean_val - 3*std_val],
                y=[0, max(hist_values)*1.2],
                mode='lines',
                name='-3σ',
                line=dict(color='orange', width=1.5, dash='dot'),
                hovertemplate=f'-3σ: {mean_val - 3*std_val:.2f}<extra></extra>'
            ))

            # 레이아웃 설정
            fig_hist.update_layout(
                title=f'{selected_var} 분포 및 합격률 (합격률: {yield_rate:.2f}%)',
                xaxis_title='값',
                yaxis_title='확률 밀도',
                hovermode='closest',
                height=500,
                showlegend=True,
                margin=dict(t=50, b=50, l=50, r=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # 그리드 추가
            fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

            # Plotly 그래프 표시
            display_plotly_centered(fig_hist)

            # 시뮬레이션을 위한 기본값 설정 (오류 수정)
            # 시뮬레이션 분포 매개변수 설정
            st.write("#### 시뮬레이션 설정")
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                sim_mean = st.slider(
                    "평균 조정",
                    min_value=float(mean_val - 3*std_val),
                    max_value=float(mean_val + 3*std_val),
                    value=float(mean_val),
                    step=float(std_val/10),
                    format="%.2f",
                    help="공정 평균값을 조정하여 시뮬레이션합니다."
                )
            
            with sim_col2:
                sim_std = st.slider(
                    "표준편차 조정",
                    min_value=float(std_val * 0.5),
                    max_value=float(std_val * 1.5),
                    value=float(std_val),
                    step=float(std_val/20),
                    format="%.2f",
                    help="공정 표준편차를 조정하여 시뮬레이션합니다."
                )
            
            # 시뮬레이션 공정능력지수 계산
            if std_val > 0 and sim_std > 0:
                # 정규성을 만족하는 경우의 공정능력지수 (시뮬레이션)
                sim_cp = (usl - lsl) / (6 * sim_std)
                sim_cpu = (usl - sim_mean) / (3 * sim_std)
                sim_cpl = (sim_mean - lsl) / (3 * sim_std)
                sim_cpk = min(sim_cpu, sim_cpl)
                
                # 정규분포 가정 하에 합격률 계산 (시뮬레이션)
                sim_z_usl = (usl - sim_mean) / sim_std
                sim_z_lsl = (lsl - sim_mean) / sim_std
                
                sim_prob_above_lsl = stats.norm.cdf(sim_z_lsl)
                sim_prob_below_usl = stats.norm.cdf(sim_z_usl)
                
                # 규격 내 비율(%) (시뮬레이션)
                sim_yield_rate = (sim_prob_below_usl - sim_prob_above_lsl) * 100
                # 불량률(PPM) (시뮬레이션)
                sim_defect_rate_ppm = (1 - (sim_prob_below_usl - sim_prob_above_lsl)) * 1000000
            else:
                st.warning("표준편차가 0입니다. 시뮬레이션을 계산할 수 없습니다.")
                sim_cp = np.nan
                sim_cpk = np.nan
                sim_yield_rate = np.nan
                sim_defect_rate_ppm = np.nan
            
            # 시뮬레이션 결과 표시
            sim_metrics_col1, sim_metrics_col2, sim_metrics_col3 = st.columns(3)
            
            with sim_metrics_col1:
                st.metric(
                    "시뮬레이션 합격률", 
                    f"{sim_yield_rate:.2f}%", 
                    delta=f"{sim_yield_rate - yield_rate:.2f}%"
                )
            
            with sim_metrics_col2:
                st.metric(
                    f"시뮬레이션 {cpk_name}", 
                    f"{sim_cpk:.2f}", 
                    delta=f"{sim_cpk - cpk:.2f}"
                )
            
            with sim_metrics_col3:
                st.metric(
                    "시뮬레이션 불량률", 
                    f"{sim_defect_rate_ppm:.0f} PPM", 
                    delta=f"{defect_rate_ppm - sim_defect_rate_ppm:.0f} PPM",
                    delta_color="inverse"
                )

            # 시뮬레이션 분포 변화 시각화를 Plotly로 변경
            # 분포 변화 시각화
            st.write("#### 분포 변화 시각화")

            # Plotly 시뮬레이션 시각화
            fig_sim = go.Figure()

            # x 범위 설정
            x_sim = np.linspace(
                min(min_val, sim_mean - 4*sim_std), 
                max(max_val, sim_mean + 4*sim_std), 
                200
            )

            # 현재 히스토그램 추가
            hist_values, hist_bins = np.histogram(var_data, bins=20, density=True)
            bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            bin_width = hist_bins[1] - hist_bins[0]

            fig_sim.add_trace(go.Bar(
                x=bin_centers,
                y=hist_values,
                width=bin_width * 0.9,
                name='현재 데이터',
                marker_color='rgba(135, 206, 235, 0.5)',
                hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
            ))

            # 현재 분포 곡선 추가
            if normality_result == "정규 분포 (p >= 0.05)":
                y_current = stats.norm.pdf(x_sim, mean_val, std_val)
                curve_name = '현재 정규분포 곡선'
            else:
                kde = gaussian_kde(var_data)
                y_current = kde(x_sim)
                curve_name = '현재 KDE 곡선'

            fig_sim.add_trace(go.Scatter(
                x=x_sim,
                y=y_current,
                mode='lines',
                name=curve_name,
                line=dict(color='blue', width=2, dash='dash'),
                hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
            ))

            # 시뮬레이션 분포 곡선 추가
            y_sim = stats.norm.pdf(x_sim, sim_mean, sim_std)
            fig_sim.add_trace(go.Scatter(
                x=x_sim,
                y=y_sim,
                mode='lines',
                name='시뮬레이션 분포',
                line=dict(color='red', width=2.5),
                hovertemplate='값: %{x:.2f}<br>밀도: %{y:.4f}<extra></extra>'
            ))

            # 규격선 추가
            fig_sim.add_trace(go.Scatter(
                x=[lsl, lsl],
                y=[0, max(max(y_current), max(y_sim), max(hist_values))*1.2],
                mode='lines',
                name='하한규격(LSL)',
                line=dict(color='purple', width=2, dash='dash'),
                hovertemplate=f'LSL: {lsl:.2f}<extra></extra>'
            ))

            fig_sim.add_trace(go.Scatter(
                x=[usl, usl],
                y=[0, max(max(y_current), max(y_sim), max(hist_values))*1.2],
                mode='lines',
                name='상한규격(USL)',
                line=dict(color='purple', width=2, dash='dash'),
                hovertemplate=f'USL: {usl:.2f}<extra></extra>'
            ))

            # 평균선 추가
            fig_sim.add_trace(go.Scatter(
                x=[mean_val, mean_val],
                y=[0, max(max(y_current), max(y_sim), max(hist_values))*1.2],
                mode='lines',
                name='현재 평균',
                line=dict(color='blue', width=2),
                hovertemplate=f'현재 평균: {mean_val:.2f}<extra></extra>'
            ))

            fig_sim.add_trace(go.Scatter(
                x=[sim_mean, sim_mean],
                y=[0, max(max(y_current), max(y_sim), max(hist_values))*1.2],
                mode='lines',
                name='시뮬레이션 평균',
                line=dict(color='red', width=2),
                hovertemplate=f'시뮬레이션 평균: {sim_mean:.2f}<extra></extra>'
            ))

            # 레이아웃 설정
            fig_sim.update_layout(
                title='시뮬레이션: 현재 분포 vs 조정된 분포',
                xaxis_title='값',
                yaxis_title='확률 밀도',
                hovermode='closest',
                height=500,
                showlegend=True,
                margin=dict(t=50, b=50, l=50, r=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # 그리드 추가
            fig_sim.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig_sim.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

            # Plotly 그래프 표시
            display_plotly_centered(fig_sim)

            # 시뮬레이션 결과 해석
            st.write("#### 시뮬레이션 결과 해석")

            # 시뮬레이션 합격률 해석
            if sim_yield_rate > yield_rate:
                st.success(f"✅ 변경된 공정 파라미터로 합격률이 {yield_rate:.2f}%에서 {sim_yield_rate:.2f}%로 {sim_yield_rate - yield_rate:.2f}% 증가했습니다.")
            else:
                st.error(f"❌ 변경된 공정 파라미터로 합격률이 {yield_rate:.2f}%에서 {sim_yield_rate:.2f}%로 {yield_rate - sim_yield_rate:.2f}% 감소했습니다.")

            # Cpk 개선 해석
            if sim_cpk > cpk:
                st.success(f"✅ {cpk_name}가 {cpk:.2f}에서 {sim_cpk:.2f}로 개선되었습니다.")
                
                if sim_cpk >= 1.33 and cpk < 1.33:
                    st.info("🎯 이 변경으로 공정능력이 '적절' 또는 '주의 필요' 수준에서 '우수' 수준으로 향상되었습니다.")
                elif sim_cpk >= 1.0 and cpk < 1.0:
                    st.info("🎯 이 변경으로 공정능력이 '부적합' 수준에서 '적절' 수준으로 향상되었습니다.")
            else:
                st.error(f"❌ {cpk_name}가 {cpk:.2f}에서 {sim_cpk:.2f}로 감소했습니다.")

            # 상세 개선사항 분석
            st.write("#### 상세 개선 분석")

            # 개선 원인 파악
            if abs(sim_mean - (usl + lsl) / 2) < abs(mean_val - (usl + lsl) / 2):
                st.write("✅ 공정 중심이 규격 중심에 더 가까워졌습니다. (중심 이탈 감소)")
            elif abs(sim_mean - (usl + lsl) / 2) > abs(mean_val - (usl + lsl) / 2):
                st.write("❌ 공정 중심이 규격 중심에서 더 멀어졌습니다. (중심 이탈 증가)")

            if sim_std < std_val:
                st.write(f"✅ 공정 산포(표준편차)가 {std_val:.2f}에서 {sim_std:.2f}로 {(1-(sim_std/std_val))*100:.1f}% 감소했습니다.")
            elif sim_std > std_val:
                st.write(f"❌ 공정 산포(표준편차)가 {std_val:.2f}에서 {sim_std:.2f}로 {((sim_std/std_val)-1)*100:.1f}% 증가했습니다.")

            # 시뮬레이션 결과를 바탕으로 한 권장 조치
            st.write("#### 권장 조치")

            if sim_cpk > cpk:
                st.write("🔍 **시뮬레이션 결과가 현재보다 개선되었습니다. 다음 조치를 고려하세요:**")
                
                if abs(sim_mean - (usl + lsl) / 2) < abs(mean_val - (usl + lsl) / 2):
                    st.write(f"1. 공정 중심을 현재 {mean_val:.2f}에서 {sim_mean:.2f}로 조정")
                    
                    # 구체적인 방법 제안
                    center_diff = sim_mean - mean_val
                    if center_diff > 0:
                        st.write(f"   - 목표값을 {center_diff:.2f} 단위 상향 조정")
                    else:
                        st.write(f"   - 목표값을 {abs(center_diff):.2f} 단위 하향 조정")
                
                if sim_std < std_val:
                    st.write(f"2. 공정 산포를 현재 {std_val:.2f}에서 {sim_std:.2f}로 감소")
                    st.write("   - 프로세스 변동 원인 분석 및 제거")
                    st.write("   - 작업자 교육 및 표준 작업 지침 개선")
                    st.write("   - 설비 안정성 향상 및 유지보수 개선")
            else:
                st.write("❌ **시뮬레이션 결과가 현재보다 악화되었습니다. 다음 사항을 고려하세요:**")
                st.write("1. 다른 매개변수 조합으로 시뮬레이션 재시도")
                
                # 최적 조건 제안
                optimal_mean = (usl + lsl) / 2
                st.write(f"2. 규격 중심({optimal_mean:.2f})에 가까운 공정 중심 설정 고려")
                st.write("3. 표준편차 감소를 위한 공정 안정화 먼저 시도")

        # --- 보고서 다운로드 섹션 시작 ---
        # st.write("--- DEBUG: Reached download section --- ") # 디버깅 제거
        st.subheader("📊 보고서 다운로드")
        st.write("분석 결과를 다운로드하여 저장하거나 공유할 수 있습니다.")
        
        # 다운로드 버튼 (st.columns 임시 제거)
        
        # CSV 다운로드 버튼
        try: # CSV 생성/버튼 오류 방지
            # st.write("--- DEBUG: Creating CSV button --- ") # 디버깅 제거
            csv_data = create_csv_data()
            csv_filename = f"{selected_var}_공정능력분석_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="CSV 파일로 다운로드",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                help="분석 결과와 데이터를 CSV 파일로 다운로드합니다. 모든 분석 결과값과 원본 데이터가 포함됩니다."
            )
            st.caption("💡 CSV 파일은 추가 분석이나 데이터 저장에 적합합니다.")
            # st.write("--- DEBUG: CSV button created successfully --- ") # 디버깅 제거
        except Exception as e:
            st.error(f"CSV 다운로드 버튼 생성 중 오류: {e}")
            # st.write(f"--- DEBUG: Error creating CSV button: {e} --- ") # 디버깅 제거
        
        # HTML 보고서 다운로드 버튼 (그래프 제외 버전)
        try: # HTML 생성/버튼 오류 방지
            # st.write("--- DEBUG: Attempting to create HTML report (No Graphs) --- ") # 디버깅 제거
            html_report = create_html_report() # 그래프 없는 버전 호출
            # st.write(f"--- DEBUG: create_html_report (No Graphs) returned ... ") # 디버깅 제거
            
            if html_report and isinstance(html_report, str) and len(html_report) > 100: 
                # st.write("--- DEBUG: HTML report (No Graphs) is valid ... ") # 디버깅 제거
                html_filename = f"{selected_var}_공정능력분석_보고서(그래프제외)_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                st.download_button(
                    label="HTML 보고서 다운로드 (그래프 제외)", # 라벨 수정
                    data=html_report,
                    file_name=html_filename,
                    mime="text/html",
                    help="분석 결과와 해석이 포함된 보고서를 HTML 형식으로 다운로드합니다. (그래프는 포함되지 않음)" # 도움말 수정
                )
                st.caption("💡 HTML 보고서는 분석 결과와 해석만 포함합니다.") # 캡션 수정
                # st.write("--- DEBUG: HTML button (No Graphs) created successfully --- ") # 디버깅 제거
            else:
                 st.warning("HTML 보고서 데이터를 생성하지 못했거나 유효하지 않습니다.")
                 # st.write(f"--- DEBUG: HTML report (No Graphs) invalid ... ") # 디버깅 제거
        except Exception as e:
            st.error(f"HTML 다운로드 버튼 생성 중 오류: {e}")
            # st.write(f"--- DEBUG: Error creating HTML button (No Graphs): {e} --- ") # 디버깅 제거
        
        # 다운로드 관련 추가 설명
        st.info("📝 HTML 보고서는 그래프 없이 분석 결과와 해석만 포함됩니다. CSV 파일은 Excel 등에서 추가 분석하려는 경우에 유용합니다.") # 설명 수정

    else: # if len(var_data) > 0: 의 else 블록
        st.error(f"선택한 변수 '{selected_var}'에 유효한 데이터가 없습니다.")
else:
    st.info("CSV 파일을 업로드해주세요. 왼쪽 사이드바에서 파일을 업로드할 수 있습니다.")


# 페이지 하단 소개
st.markdown("---")


st.markdown("**문의 및 피드백:**")
st.error("문제점 및 개선요청사항이 있다면, 정보기획팀 고동현 주임(내선: 189)에게 피드백 부탁드립니다. 지속적인 개선에 반영하겠습니다. ")