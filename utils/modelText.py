import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # Si deseas guardar/cargar el tokenizer
import re
import os

# Carga el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__),'model_2_texto.h5') # Cambia el número de versión si es necesario
model = load_model(model_path)

# Etiquetas de emociones (asegúrate de que coincidan con las usadas en el entrenamiento)
EMOTION_LABELS = {
    0: 'sadness',
    1: 'happiness',
    2: 'love',
    3: 'anger',
    4: 'worry',
    5: 'neutral',
}

# Carga el tokenizer (suponiendo que lo guardaste en un archivo)
tokenizer_path = os.path.join(os.path.dirname(__file__),'tokenizer1.pkl')  # Cambia el nombre si usaste otro
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Configuración del padding
MAX_LEN = 120  # Debe coincidir con el valor usado en el entrenamiento

# Función para limpiar el texto
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        text = text.lower().strip()
        return text
    return ""

# Función para predecir la emoción principal
def predict_text(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]
    max_index = np.argmax(prediction)
    main_emotion = EMOTION_LABELS[max_index]
    confidence = prediction[max_index]
    return str(main_emotion)

# Ejemplo de predicción7
sample_text = 'I love my dog'
emotion = predict_text(sample_text)
print(f"La emoción principal es: {emotion}")