import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec

nltk.download('punkt')

def train_word2vec_on_sentence(sentence, vector_size=10, window=3):
    # Tokenize the sentence
    tokenized_sentence = [word_tokenize(sentence.lower())]
    # Train Word2Vec model
    model_w2v = Word2Vec(sentences=tokenized_sentence, vector_size=vector_size, window=window, min_count=1)
    return model_w2v

# Convert sentence to a vector using Word2Vec
def sentence_to_vector(sentence, model_w2v, vector_size=10):
    tokenized_sentence = word_tokenize(sentence.lower())
    vectors = [model_w2v.wv[word] for word in tokenized_sentence if word in model_w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# Make a prediction with the emotion model
def predict_emotion(text, emotion_model, model_w2v):
    # Convert the sentence to a vector
    sentence_vector = sentence_to_vector(text, model_w2v)

    # If no vector is generated, return a message
    if not sentence_vector.any():
        return "No relevant words found in the sentence."

    # Predict the emotion
    prediction = emotion_model.predict(sentence_vector.reshape(1, -1))
    emotion_labels = {0: 'sadness', 1: 'happiness', 2: 'love', 3: 'anger', 4: 'worry', 5: 'neutral'}
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion

def predict_text(text_input):

    emotion_model = load_model('./model1.h5')

    emotion_labels = {
        0: 'sadness',
        1: 'happiness',
        2: 'love',
        3: 'anger',
        4: 'worry',
        5: 'neutral'
    }


    model_w2v = train_word2vec_on_sentence(text_input)

    # Make a prediction
    result = predict_emotion(text_input, emotion_model, model_w2v)

    print(result)
    return str(result)

# if __name__=='__main__':
#     predict_text("im soooooo sad, kill me please")