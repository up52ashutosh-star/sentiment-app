import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

st.title("Sentiment Analysis App")

text = st.text_input("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = model(text)
        st.write(result)
