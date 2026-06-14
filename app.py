def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

import streamlit as st
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download only once
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

st.title("Aspect Based Sentiment Analysis (Lightweight)")

analyzer = SentimentIntensityAnalyzer()

def extract_aspects(tokens, tags):
    # simple noun extraction
    aspects = [word for word, tag in zip(tokens, tags) if tag.startswith("NN")]
    return aspects

def analyze_text(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    output = ""

    for sent in sentences:
        tokens = word_tokenize(sent)
        tags = pos_tag(tokens)

        aspects = [w for w, t in zip(tokens, [t[1] for t in tags]) if t.startswith("NN")]

        score = analyzer.polarity_scores(sent)["compound"]

        if score >= 0.05:
            label = "POSITIVE 😊"
        elif score <= -0.05:
            label = "NEGATIVE 😞"
        else:
            label = "NEUTRAL 😐"

        output += f"""
----------------------------------------
Review Part:
{sent}

POS Tags:
{tags}

Aspects:
{aspects}

Sentiment:
{label}

Score:
{score:.4f}
"""

    return output
----------------------------------------
Review Part:
{sent}

POS Tags:
{tags}

Aspects (Nouns):
{aspects}

Sentiment Label:
{label} {emoji}

Score:
{compound}

Confidence:
{abs(compound)*100:.2f}%
"""
        result_blocks.append(block)

    return "\n".join(result_blocks)


text = st.text_area("Enter Review Text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        output = analyze_text(text)
        st.text(output)
