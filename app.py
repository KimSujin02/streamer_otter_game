import base64
import re
import html
from typing import Dict, Tuple

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import base64

st.set_page_config(
    page_title="방송인 키우기 게임",
    page_icon="📺",
    layout="wide",
)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def get_state_name(stress: int) -> str:
    if stress <= 0:
        return "압도적 행복"
    if stress <= 10:
        return "행복"
    if stress <= 20:
        return "기쁨"
    if stress <= 50:
        return "평온"
    if stress <= 70:
        return "당황"
    if stress <= 80:
        return "슬픔"
    return "멘붕"

def get_state_image(stress: int) -> str:
    if stress <= 0:
        return "./images/stress_0.PNG"
    if stress <= 10:
        return "./images/stress_1_10.PNG"
    if stress <= 20:
        return "./images/stress_11_20.PNG"
    if stress <= 50:
        return "./images/stress_21_50.PNG"
    if stress <= 70:
        return "./images/stress_51_70.PNG"
    if stress <= 80:
        return "./images/stress_71_90.PNG"
    return "./images/stress_91_100.PNG"


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_avatar_html(stress: int) -> str:
    image_path = get_state_image(stress)
    state = get_state_name(stress)
    img_base64 = image_to_base64(image_path)

    return f"""
    <div style="
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border-radius: 24px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.06);
    ">
        <img 
            src="data:image/png;base64,{img_base64}" 
            alt="{state}" 
            style="
                width: 100%;
                max-width: 420px;
                height: auto;
                border-radius: 18px;
            "
        />
        <div style="
            margin-top: 12px;
            font-size: 26px;
            font-weight: 700;
            color: #333;
        ">
            수달군
        </div>
        <div style="
            margin-top: 6px;
            font-size: 18px;
            color: #555;
        ">
            현재 상태: <b>{state}</b>
        </div>
    </div>
    """

# 허깅 페이스 파이프라인
@st.cache_resource(show_spinner=False)
def load_pipelines():
    sentiment_pipe = pipeline(
        "text-classification",
        model="daekeun-ml/koelectra-small-v3-nsmc",
        tokenizer="daekeun-ml/koelectra-small-v3-nsmc",
        truncation=True,
    )

    toxic_pipe = pipeline(
        "text-classification",
        model="jinkyeongk/kcELECTRA-toxic-detector",
        tokenizer="jinkyeongk/kcELECTRA-toxic-detector",
        truncation=True,
    )

    gen_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    return sentiment_pipe, toxic_pipe, gen_tokenizer, gen_model


# =========================
# 감정 분류
# =========================
def map_sentiment(result: Dict) -> Tuple[str, float]:
    label = str(result.get("label", "")).lower().strip()
    score = float(result.get("score", 0.0))

    if label in {"label_1", "1", "positive", "pos"}:
        return "positive", score
    if label in {"label_0", "0", "negative", "neg"}:
        return "negative", score

    return "neutral", score


def map_toxic(result: Dict) -> Tuple[str, float]:
    label = str(result.get("label", "")).lower().strip()
    score = float(result.get("score", 0.0))

    toxic_tokens = ["toxic", "hate", "offensive", "혐오", "악성"]
    safe_tokens = ["safe", "clean", "normal", "비혐오", "일반"]

    if any(tok in label for tok in toxic_tokens):
        return "toxic", score
    if any(tok in label for tok in safe_tokens):
        return "safe", score

    # Model card examples often use label_1=toxic, label_0=safe
    if label in {"label_1", "1"}:
        return "toxic", score
    if label in {"label_0", "0"}:
        return "safe", score

    return "safe", score

# 스트레스 계산
def calc_stress_delta(sentiment_label: str, sentiment_score: float, toxic_label: str, toxic_score: float) -> int:
    delta = 0

    if sentiment_label == "positive":
        delta -= 10 if sentiment_score >= 0.8 else 6
    elif sentiment_label == "negative":
        delta += 10 if sentiment_score >= 0.8 else 6

    if toxic_label == "toxic":
        delta += 15 if toxic_score >= 0.90 else 12

    return delta


def build_prompt(comment: str, state: str):
    return [
        {
            "role": "system",
            "content": "한국어로 짧게 반응하는 애교많은 귀여운 방송인이다.",
        },
        {
            "role": "user",
            "content": f"상태:{state} 댓글:{comment} 반응 한 줄:",
        },
    ]

def get_fallback_reply(state: str) -> str:
    fallback = {
        "압도적 행복": "오늘 너무 행복해용!!! 💖",
        "행복": "와아~ 너무 좋아요! 헤헤 💖",
        "기쁨": "좋은 말 들으니까 힘나요!",
        "평온": "좋은 댓글 감사합니다~",
        "당황": "앗, 조금 떨리네요…",
        "슬픔": "우우… 그래도 힘내볼게요…",
        "멘붕": "흑… 그래도 방송은 계속할게요…",
    }
    return fallback.get(state, "고마워요!")

def postprocess_reply(text: str, fallback_state: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"' ")

    # 줄바꿈만 자르기
    if "\n" in text:
        text = text.split("\n")[0].strip()

    if not text:
        return get_fallback_reply(fallback_state)

    if "�" in text or "Ã" in text or "ð" in text:
        return get_fallback_reply(fallback_state)

    if not re.search(r"[가-힣]", text):
        return get_fallback_reply(fallback_state)

    return text


def init_state():
    defaults = {
        "stress": 30,
        "reply": "안녕하세요~ 오늘 방송도 잘 부탁드려요!",
        "comments": [],
        "logs": [],
        "subscriber": 100,
        "day": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# UI
init_state()

st.title("📺 방송하는 수달 댓글 게임")
st.caption("댓글에 따라 스트레스가 변하고, 수달군이 반응합니다!")

with st.sidebar:
    if st.button("게임 초기화", use_container_width=True):
        for key in ["stress", "reply", "comments", "logs", "subscriber", "day"]:
            st.session_state.pop(key, None)
        st.rerun()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown(make_avatar_html(st.session_state.stress), unsafe_allow_html=True)
    st.progress(st.session_state.stress / 100)
    st.write(f"**스트레스:** {st.session_state.stress} / 100")

with right:
    st.subheader("현재 대사")
    st.info(st.session_state.reply)

    comment = st.text_input("댓글 입력", placeholder="예: 오늘 방송 너무 재밌어요!", key="comment_input")

    col_a, col_b = st.columns([1, 1])
    submit = col_a.button("댓글 등록", use_container_width=True)
    
    st.subheader("최근 댓글")
    if not st.session_state.comments:
        st.write("아직 댓글이 없습니다.")
    else:
        for idx, c in enumerate(st.session_state.comments[:10], start=1):
            st.markdown(f"**{idx}.** {html.escape(c)}")
    
    if submit:
        user_comment = clean_text(comment)

        if not user_comment:
            st.warning("댓글을 입력해 주세요.")
        else:
            try:
                with st.spinner("모델 불러와서 분석 중입니다..."):
                    sentiment_pipe, toxic_pipe, gen_tokenizer, gen_model = load_pipelines()

                sentiment_raw = sentiment_pipe(user_comment)[0]
                toxic_raw = toxic_pipe(user_comment)[0]
                
                sentiment_label, sentiment_score = map_sentiment(sentiment_raw)
                toxic_label, toxic_score = map_toxic(toxic_raw)

                delta = calc_stress_delta(sentiment_label, sentiment_score, toxic_label, toxic_score)
                st.session_state.stress = clamp(st.session_state.stress + delta)
                state_name = get_state_name(st.session_state.stress)

                messages = build_prompt(user_comment, state_name)

                text = gen_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = gen_tokenizer([text], return_tensors="pt").to(gen_model.device)

                with torch.inference_mode():
                    generated_ids = gen_model.generate(
                        **model_inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        top_p=0.8,
                        temperature=0.6,
                        repetition_penalty=1.2,
                        pad_token_id=gen_tokenizer.eos_token_id,
                    )

                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                generated = gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                print("생성된 텍스트:", generated)
                reply = postprocess_reply(generated, state_name)

                st.session_state.reply = reply
                st.session_state.comments.insert(0, user_comment)
                st.session_state.logs.insert(
                    0,
                    {
                        "comment": user_comment,
                        "sentiment": sentiment_label,
                        "sentiment_score": sentiment_score,
                        "toxic": toxic_label,
                        "toxic_score": toxic_score,
                        "delta": delta,
                        "stress": st.session_state.stress,
                        "reply": reply,
                    },
                )
                st.rerun()

            except Exception as exc:
                st.error(
                    "모델 실행 중 오류가 발생했습니다. 처음 실행 시 모델 다운로드 때문에 시간이 걸릴 수 있습니다.\n\n"
                    f"오류: {exc}"
                )

st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("판정 로그")
    if not st.session_state.logs:
        st.write("아직 로그가 없습니다.")
    else:
        for log in st.session_state.logs[:10]:
            sentiment_txt = f"{log['sentiment']} ({log['sentiment_score']:.2f})"
            toxic_txt = f"{log['toxic']} ({log['toxic_score']:.2f})"
            delta_txt = f"{log['delta']:+d}"
            st.markdown(
                f"""
                <div style="padding:12px 14px; border-radius:14px; border:1px solid #e5e7eb; margin-bottom:10px;">
                    <div><b>댓글</b>: {html.escape(log['comment'])}</div>
                    <div><b>감정</b>: {sentiment_txt}</div>
                    <div><b>독성</b>: {toxic_txt}</div>
                    <div><b>스트레스 변화</b>: {delta_txt} / 현재 {log['stress']}</div>
                    <div><b>대사</b>: {html.escape(log['reply'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
with col2:
    pass