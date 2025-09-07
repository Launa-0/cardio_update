import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# ✅ Gemini API 키 설정 (https://makersuite.google.com/app 에서 발급 가능)
genai.configure(api_key="AIzaSyAQohnRRkfkfagrNtxBJvaIjepMiNjfLdM")  # ← 본인의 키로 교체하세요

# ✅ Gemini 모델 초기화
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ 변수명 한글 매핑
feature_names_ko = {
    'age': '나이 (세)',
    'gender': '성별',
    'ap_hi': '수축기 혈압',
    'ap_lo': '이완기 혈압',
    'cholesterol': '콜레스테롤 등급',
    'gluc': '혈당 등급',
    'smoke': '흡연 여부',
    'alco': '음주 여부',
    'active': '운동 여부',
    'BMI': '체질량지수'
}

def translate_value(feature, value):
    if feature == "cholesterol":
        return ['안전', '양호', '위험'][int(value) - 1]
    elif feature == "gluc":
        return ['안전', '양호', '위험'][int(value) - 1]
    elif feature == "gender":
        return "남성" if value == 1 else "여성"
    elif feature == "smoke":
        return "흡연" if value == 1 else "비흡연"
    elif feature == "alco":
        return "음주" if value == 1 else "비음주"
    elif feature == "active":
        return "운동함" if value == 1 else "운동 안함"
    elif feature == "age":
        return f"{int(value // 365)}세"
    elif feature == "BMI":
        return f"{value:.1f}"
    else:
        return value

# ✅ 모델 불러오기
@st.cache_resource
def load_model():
    with open("xgb_best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ✅ Streamlit UI
st.set_page_config(page_title="심혈관 위험 예측기", layout="wide")
st.title("💓 당신의 심혈관 건강은 안전한가요?")

st.markdown("""
**XGBoost 기반 심혈관 질환 위험 예측 & 건강 개선 제안**  
사용자의 건강 정보를 바탕으로 위험도를 예측하고,  
개인별로 중요한 위험 요인을 설명하며 건강 개선 제안도 드립니다.
""")

# --- 입력 ---
st.sidebar.header("📝 건강 정보 입력")
age = st.sidebar.slider("나이", 20, 80, 60)
gender = st.sidebar.radio("성별", ["남성", "여성"])
height = st.sidebar.number_input("키 (cm)", 140, 200, 170)
weight = st.sidebar.number_input("몸무게 (kg)", 40, 150, 65)
ap_hi = st.sidebar.slider("수축기 혈압", 90, 200, 120)
ap_lo = st.sidebar.slider("이완기 혈압", 40, 130, 80)
cholesterol = st.sidebar.selectbox("콜레스테롤 등급", ["안전", "양호", "위험"])
gluc = st.sidebar.selectbox("혈당 등급", ["안전", "양호", "위험"])
smoke = st.sidebar.checkbox("흡연 여부")
alco = st.sidebar.checkbox("음주 여부")
active = st.sidebar.checkbox("운동을 규칙적으로 하나요?")

cholesterol_map = {"안전": 1, "양호": 2, "위험": 3}
gluc_map = {"안전": 1, "양호": 2, "위험": 3}

# --- 입력값 구성 ---
bmi = weight / ((height / 100) ** 2)
input_data = {
    'age': age * 365,
    'gender': 1 if gender == "남성" else 2,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol_map[cholesterol],
    'gluc': gluc_map[gluc],
    'smoke': int(smoke),
    'alco': int(alco),
    'active': int(active),
    'BMI': bmi
}
input_df = pd.DataFrame([input_data])

# ✅ 예측 결과
st.subheader("📊 예측 결과")
proba = model.predict_proba(input_df)[0][1]
st.metric(label="심혈관 질환 위험도", value=f"{proba * 100:.2f}%")

# ✅ SHAP 기반 설명
st.markdown("#### 📌 예측 근거 (개인별 변수 기여도 기준)")
try:
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "변수": input_df.columns,
        "입력값": [translate_value(col, val) for col, val in input_df.iloc[0].items()],
        "기여도": shap_values.values[0]
    })
    shap_df["기여도절댓값"] = shap_df["기여도"].abs()
    shap_df["기여도비율(%)"] = shap_df["기여도절댓값"] / shap_df["기여도절댓값"].sum() * 100
    shap_df = shap_df.sort_values(by="기여도절댓값", ascending=False).head(3)

    for _, row in shap_df.iterrows():
        sign = "높였습니다" if row["기여도"] > 0 else "낮췄습니다"
        st.markdown(f"""
        • **{row['변수']}** 값이 **{row['입력값']}**로 입력되어, 이로 인해 심혈관 위험 예측 확률을 **{abs(row['기여도']):.3f}만큼 {sign}**.  
        전체 영향 중 약 {row["기여도비율(%)"]:.1f}%를 차지했습니다.
        """, unsafe_allow_html=True)

    with st.expander("🔍 전체 변수 영향 보기 (SHAP 방향성 시각화)"):
        shap_df_full = pd.DataFrame({
            "변수": input_df.columns,
            "입력값": [translate_value(col, val) for col, val in input_df.iloc[0].items()],
            "기여도": shap_values.values[0]
        })

        shap_df_full = shap_df_full.sort_values(by="기여도", key=np.abs, ascending=False).head(10)

        # 색상 지정: 양수(빨간색), 음수(파란색)
        colors = ['red' if val > 0 else 'blue' for val in shap_df_full["기여도"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(
            shap_df_full["변수"],
            shap_df_full["기여도"],
            color=colors,
            edgecolor='black'
        )

        # 기준선
        ax.axvline(0, color='gray', linewidth=1)

        # 축 및 제목 설정
        ax.set_xlabel("SHAP Value")
        ax.set_title("SHAP Value Contribution (Red = Increase ↑ / Blue = Decrease ↓)")
        ax.invert_yaxis()

        # x축 범위 조절 (기여도의 최대 절댓값 기준으로 약간 여유 있게 설정)
        max_val = max(abs(shap_df_full["기여도"].max()), abs(shap_df_full["기여도"].min()))
        ax.set_xlim(-max_val * 1.2, max_val * 1.2)

        # 텍스트 표시 (양수면 + 붙이기)
        for bar, val in zip(bars, shap_df_full["기여도"]):
            text = f"{val:.3f}" if val < 0 else f"+{val:.3f}"
            ax.text(val + 0.02 if val > 0 else val - 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    text,
                    va='center',
                    ha='left' if val > 0 else 'right',
                    fontsize=10,
                    color='black')

        # 레이아웃 자동 조정
        plt.tight_layout()
        st.pyplot(fig)



except Exception as e:
    st.warning("SHAP 값을 계산하는 중 오류가 발생했습니다.")
    st.exception(e)

# ✅ Gemini 개선 제안
st.subheader("🛠️ 개선 제안 (Gemini 기반)")
try:
    # 프롬프트용 출력값만 가공
    readable_input = {k: translate_value(k, v) for k, v in input_data.items()}

    prompt = (
        "당신은 건강 전문가입니다. 다음은 사용자의 건강 정보입니다:\n\n"
        f"{readable_input}\n\n"
        f"이 사용자의 심혈관 위험도는 {proba * 100:.2f}%입니다.\n"
        "이 정보를 기반으로 심혈관 건강을 개선할 수 있는 생활습관 변화나 조언을 한국어로 3~5개 정도 제시해 주세요."
    )
    response = gemini_model.generate_content(prompt)
    st.markdown(response.text)
except Exception as e:
    st.warning("개선 제안 생성 중 오류가 발생했습니다.")
    st.exception(e)

# ✅ 시뮬레이터
st.subheader("⚙️ 시뮬레이터: 혈압 조정 시 위험도 변화")
sim_ap_hi = st.slider("수축기 혈압 (mmHg)", 90, 200, ap_hi)
sim_ap_lo = st.slider("이완기 혈압 (mmHg)", 40, 130, ap_lo)

sim_data = input_data.copy()
sim_data['ap_hi'] = sim_ap_hi
sim_data['ap_lo'] = sim_ap_lo
sim_df = pd.DataFrame([sim_data])
sim_proba = model.predict_proba(sim_df)[0][1]

st.info(f"혈압을 {ap_hi}/{ap_lo} → {sim_ap_hi}/{sim_ap_lo} mmHg로 조정하면, 위험도는 {sim_proba*100:.2f}%로 바뀝니다.")
