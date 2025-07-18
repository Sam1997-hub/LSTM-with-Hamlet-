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
    
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Keep last n tokens
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Get scalar value

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "No match found"

# Streamlit UI
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter a sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]  # model.input_shape returns (None, max_sequence_len)
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"**Next word:** `{next_word}`")
