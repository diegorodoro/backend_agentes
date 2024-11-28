import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize


def predict_text(text):

    emotion_model = load_model('./model1.h5')

    emotion_labels = {
        0: 'sadness',
        1: 'happiness',
        2: 'love',
        3: 'anger',
        4: 'worry',
        5: 'neutral'
    }

    pass

if __name__=='__main__':
    pass