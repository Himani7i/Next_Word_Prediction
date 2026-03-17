import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_prediction_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction")
input_text = st.text_input("Enter a phrase:")

if st.button("Predict Next Word"):
    if input_text:
        max_sequence_len = model.input_shape[1]+1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Predicted next word: {next_word}")
    else:
        st.write("Please enter a phrase to predict the next word.")