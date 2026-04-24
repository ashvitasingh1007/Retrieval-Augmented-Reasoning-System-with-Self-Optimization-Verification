import streamlit as st
import base64
import time
from dotenv import load_dotenv

from retriever import get_retriever
from generator import generate_answer
from verifier import verify_answer
from evaluator import evaluate

load_dotenv()

# -------------------------------
# 📌 SAMPLE QUESTIONS
# -------------------------------
sample_queries = [
    "What is retrieval augmented generation and why is it important?",
    "How does verification reduce hallucination in AI systems?",
    "Explain hybrid retrieval (dense + keyword) in RAG",
    "What are the limitations of RAG systems?",
]

# -------------------------------
# 🖼️ BACKGROUND IMAGE HELPER
# -------------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image("assets/background.png")

# -------------------------------
# ⚙️ PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Peebite AI", layout="wide")

# -------------------------------
# 🎨 CUSTOM CSS
# -------------------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/png;base64,{img_base64}") no-repeat center center fixed;
    background-size: cover;
}}

.block-container {{
    background-color: rgba(15, 23, 42, 0.88);
    padding: 2rem;
    border-radius: 12px;
    backdrop-filter: blur(6px);
}}

h1, h2, h3 {{
    color: #38bdf8;
}}

.stTextInput input {{
    background-color: rgba(30, 41, 59, 0.9);
    color: white;
}}

.card {{
    background-color: rgba(30, 41, 59, 0.9);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 HEADER
# -------------------------------
st.title("🔍 Peebite — Retrieval-Augmented Reasoning System")
st.markdown("AI system that **retrieves, reasons, verifies, and evaluates** answers.")

# -------------------------------
# 📊 SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Suggested Questions")

# Session state
if "query" not in st.session_state:
    st.session_state.query = ""

# Sidebar buttons
for q in sample_queries:
    short_q = q[:60] + "..." if len(q) > 60 else q
    if st.sidebar.button(short_q):
        st.session_state.query = q
        st.rerun()

# -------------------------------
# 🔎 LOAD RETRIEVER
# -------------------------------
retriever, docs = get_retriever()

# -------------------------------
# 📝 INPUT
# -------------------------------
query = st.text_input("Enter your question:", value=st.session_state.query)

# -------------------------------
# 🚀 RUN SYSTEM
# -------------------------------
if query:

    # -------------------------------
    # ⚙️ LIVE STATUS
    # -------------------------------
    st.markdown("### ⚙️ System Status")

    status_placeholder = st.empty()

    with status_placeholder.container():
        st.write("🔹 Initializing system...")
        time.sleep(0.3)
        st.write("🔹 Retrieving knowledge...")
        time.sleep(0.3)
        st.write("🔹 Generating answer...")
        time.sleep(0.3)
        st.write("🔹 Verifying output...")
        time.sleep(0.3)
        st.write("🔹 Evaluating confidence...")
        time.sleep(0.3)

    # -------------------------------
    # 🧠 CORE PIPELINE
    # -------------------------------
    with st.spinner("Running reasoning pipeline..."):

        # Retrieve
        dense_docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in dense_docs])

        # Generate
        answer = generate_answer(query, context)

        # Verify
        verification = verify_answer(query, answer, context)

        # Evaluate
        score = evaluate(answer, context)

    status_placeholder.empty()

    # -------------------------------
    # 📊 SCORE PARSE
    # -------------------------------
    try:
        score_value = float(score)
    except:
        score_value = None

    # -------------------------------
    # 🧠 OUTPUT TABS
    # -------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧠 Answer", "🔍 Verification", "📊 Evaluation", "📄 Context"]
    )

    # ANSWER
    with tab1:
        st.markdown("### Generated Answer")
        st.markdown(f"<div class='card'>{answer}</div>", unsafe_allow_html=True)

    # VERIFICATION
    with tab2:
        st.markdown("### Verification Result")
        st.markdown(f"<div class='card'>{verification}</div>", unsafe_allow_html=True)

    # EVALUATION
    with tab3:
        st.markdown("### Confidence Score")

        if score_value is not None:
            st.progress(score_value)

            if score_value > 0.8:
                st.success("High confidence answer")
            elif score_value > 0.5:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence - needs improvement")
        else:
            st.write(score)

    # CONTEXT
    with tab4:
        st.markdown("### Retrieved Context (Transparency)")
        st.markdown(f"<div class='card'>{context}</div>", unsafe_allow_html=True)

    # -------------------------------
    # 📜 EXECUTION TIMELINE
    # -------------------------------
    with st.expander("📜 Execution Timeline"):
        st.write("1. Query received")
        st.write("2. Retriever gathered relevant knowledge")
        st.write("3. Generator produced structured answer")
        st.write("4. Verifier checked factual grounding")
        st.write("5. Evaluator scored reliability")

    # -------------------------------
    # 🧠 SYSTEM VALUE
    # -------------------------------
    st.markdown("### 🧠 Why Peebite Matters")
    st.markdown("""
Peebite is not just a question-answering system.

It demonstrates:
- Retrieval-grounded reasoning  
- Verification-based hallucination control  
- Confidence scoring for reliability  
- Transparent AI decision pipeline  

Designed to simulate how a real researcher validates answers.
""")