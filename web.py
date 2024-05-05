import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = tf.keras.models.load_model('hate_speech_model.h5')

tokenizer_file_path = "tokenizer.pkl"

# Deserialize and load the Tokenizer object using pickle
with open(tokenizer_file_path, 'rb') as file:
    tokenizer = pickle.load(file)

# Function for preprocessing input text
def preprocess_text(text):
    # Tokenization
    sequences = tokenizer.texts_to_sequences(text)
    # Padding sequences
    data = pad_sequences(sequences, maxlen=100,padding='post',truncating='post')
    return data

# Function for predicting hate speech
def predict_hate_speech(text):
    preprocessed_text = preprocess_text([text])
    prediction = model.predict(preprocessed_text)
    if prediction[0] > prediction[1]:
        if prediction[0] > prediction[2]:
            return 0
        else:
            return 2
    else:
        if prediction[1] > prediction[2] :
            return 1
        else:
            return 2

    return prediction

# Streamlit app
def main():
    st.title("Hate Speech Detection App")
    input_text = st.text_input("Enter text:")
    if st.button("Check for Hate Speech"):
        prediction = predict_hate_speech(input_text)
        if prediction == 0:
            st.error("Hate speech detected!")
        elif prediction == 1:
            st.success("offensive speech detected.")
        else:
            st.success("normal speech detected.")

if __name__ == "__main__":
    main()
