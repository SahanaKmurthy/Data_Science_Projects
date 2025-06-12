import streamlit as st 
import numpy as np 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the LSTM model
model = load_model("next_word_lstm.h5")

## Load the tokenizer
with open('tokernizer.pkl' , 'rb') as file:
    tokenizer = pickle.load(file)
    
## Function to predict the next words
def predict_next_word(model, tokenizer, text, max_sequence_len):
    tokens_list = tokenizer.texts_to_sequences([text])[0]
    if len(tokens_list) > max_sequence_len:
        tokens_list = tokens_list[-(max_sequence_len-1) : ] # ensure the sequence length matches the max sequence length
    token_list = pad_sequences([tokens_list] , maxlen= max_sequence_len-1, padding= 'pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word , index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


st.title("Next word prediction with LSTM")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")
