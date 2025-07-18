import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model ---
@st.cache_resource
def load_saved_model():
    return tf.keras.models.load_model('next_word_saved_model', compile=False)

model = load_saved_model()

# --- Load Tokenizer ---
@st.cache_resource
def load_tokenizer():
    with open('tokenizer_config.json', 'r') as f:
        tokenizer_json = f.read()
        return tokenizer_from_json(tokenizer_json)

tokenizer = load_tokenizer()

# --- Prediction Function ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Clean input
    text = text.lower().strip()

    # Convert text to token sequence
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return "Input contains unknown words"

    # Trim/pad to match input size
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Use SavedModel signature to make prediction
    infer = model.signatures["serving_default"]
    input_name = list(infer.structured_input_signature[1].keys())[0]
    input_tensor = tf.convert_to_tensor(token_list, dtype=tf.int32)
    outputs = infer(**{input_name: input_tensor})

    # Extract and decode predicted word index
    predicted = list(outputs.values())[0].numpy()
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "No match found"

# --- Streamlit UI ---
st.title("📚 Next Word Predictor (LSTM + Hamlet)")

input_text = st.text_input("Enter a sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    # Optionally extract sequence length from model input
    infer = model.signatures["serving_default"]
    input_tensor_info = list(infer.structured_input_signature[1].values())[0]
    max_sequence_len = input_tensor_info.shape[1] + 1  # Add 1 for target token

    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"**Input:** `{input_text}`")
    st.success(f"**Predicted Next Word:** `{next_word}`")
