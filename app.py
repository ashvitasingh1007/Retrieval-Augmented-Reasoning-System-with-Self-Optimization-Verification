import streamlit as st
from dotenv import load_dotenv

from retriever import get_retriever
from generator import generate_answer
from verifier import verify_answer
from evaluator import evaluate

load_dotenv()

st.title("Retrieval-Augmented Reasoning System")

# Load retriever once
retriever, docs = get_retriever()

query = st.text_input("Enter your question")

if query:
    # Retrieve context
    dense_docs = retriever.invoke(query)
    dense_context = " ".join([doc.page_content for doc in dense_docs])

    context = dense_context

    # Generate answer
    answer = generate_answer(query, context)

    # Verify
    verification = verify_answer(query, answer, context)

    # Evaluate
    score = evaluate(answer, context)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Verification")
    st.write(verification)

    st.markdown("### Score")
    st.write(score)

    st.markdown("### Retrieved Context (for transparency)")
    st.write(context)
    