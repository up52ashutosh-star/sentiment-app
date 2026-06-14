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
    sentences = sent_tokenize(text)
    result_blocks = []

    for sent in sentences:
        tokens = word_tokenize(sent)
        tags = pos_tag(tokens)

        aspects = extract_aspects(tokens, [t[1] for t in tags])

        score = analyzer.polarity_scores(sent)
        compound = score["compound"]

        if compound >= 0.05:
            label = "POSITIVE"
            emoji = "😊"
        elif compound <= -0.05:
            label = "NEGATIVE"
            emoji = "😞"
        else:
            label = "NEUTRAL"
            emoji = "😐"

        block = f"""
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
