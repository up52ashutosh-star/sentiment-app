import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sent_pipeline = load_model()

st.title("Sentiment Analysis App")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        result = sent_pipeline(review)[0]

        st.write("Sentiment:", result["label"])
        st.write("Confidence:", f"{result['score']*100:.2f}%")
