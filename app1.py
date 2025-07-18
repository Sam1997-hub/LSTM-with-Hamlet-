import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model (do NOT compile, avoids legacy issues)
model = load_model('next_word_saved_model', compile=False)

with open('tokenizer_config.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    
# Predict the next word
def predict_next_word(input_text):
    max_sequence_len = model.input_shape[1]  # should match what you trained with

    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    
    # Get word from index
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# Streamlit UI
st.title("Next Word Prediction (LSTM + Hamlet)")

input_text = st.text_input("Enter a phrase:")

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter a phrase.")
    else:
        next_word = predict_next_word(input_text)
        st.write(f"**Next word prediction:** `{next_word}`")

