import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# Preprocess text input
import re

def preprocess_text(text):
    words = re.findall(r"\b\w+\b", text.lower())  # Tokenize and remove punctuation
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is OOV
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Prediction Score:** {score:.6f}")
else:
    st.write("Please enter a movie review.")