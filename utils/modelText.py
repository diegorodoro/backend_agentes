from nltk.corpus import stopwords
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import numpy as np
import re
import pickle

import os

nltk.download('stopwords')

# Carga el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__),'model_2_texto.h5') # Cambia el número de versión si es necesario

model = load_model(model_path)

# Etiquetas de emociones (asegúrate de que coincidan con las usadas en el entrenamiento)
EMOTION_LABELS = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}

# Carga el tokenizer (suponiendo que lo guardaste en un archivo)
tokenizer_path = os.path.join(os.path.dirname(__file__),'tokenizer.pickle')  # Cambia el nombre si usaste otro
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Definir las stopwords
    words = text.split()  # Dividir el texto en palabras
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Filtrar stopwords
    return " ".join(filtered_words)

def clean_punctuation(text):
    # Eliminar signos de puntuación, dejando solo letras y espacios
    return re.sub(r'[^\w\s]', '', text)

# Función para predecir la emoción principal
def predict_text(text):
    cleaned_sentences = [clean_punctuation(remove_stopwords(text))]
    sequences = tokenizer.texts_to_sequences(cleaned_sentences)
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    emotion= EMOTION_LABELS[np.argmax(prediction)]
    return str(emotion)

# Ejemplo de predicción7
sample_text = 'Im so happy i passed quarentine'
emotion = predict_text(sample_text)
print(f"La emoción principal es: {emotion}")