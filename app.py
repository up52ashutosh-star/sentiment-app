import streamlit as st
from transformers import pipeline

# Load model only once (IMPORTANT FOR MEMORY)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

st.title("Sentiment Analysis App")

text = st.text_input("Enter text")

if st.button("Analyze"):
    result = model(text)
    st.write(result)
