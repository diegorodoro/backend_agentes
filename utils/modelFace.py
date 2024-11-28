import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Función para preprocesar imágenes desde un arreglo
def preprocess_image_from_array(image, target_size=(256, 256)):
    img = cv2.resize(image, target_size)  # Redimensionar la imagen
    img_array = img_to_array(img)  # Convertir a arreglo
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones
    img_array = img_array / 255.0  # Normalizar
    return img_array

# Función para detectar y recortar el rostro
def detect_and_crop_face(image, yolo_model, padding=0.15):
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Obtener las coordenadas de las cajas
    confidences = results[0].boxes.conf.cpu().numpy().tolist()  # Obtener las confianzas

    if len(boxes) > 0:
        max_index = confidences.index(max(confidences))  # Índice de mayor confianza
        x1, y1, x2, y2 = map(int, boxes[max_index])  # Coordenadas de la caja

        # Calcular el tamaño del padding
        h, w, _ = image.shape
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * padding)  # Padding horizontal
        pad_y = int(box_height * padding)  # Padding vertical

        # Ajustar las coordenadas con padding, asegurándose de no salir de la imagen
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Recortar la cara con padding
        cropped_face = image[y1:y2, x1:x2]
        return cropped_face, max(confidences)
    
    return None, 0  # No se detectó ninguna cara


# Función principal
def main():
    # Carga del modelo YOLO para detección de caras
    yolo_model = YOLO("./yolov8m-face.pt")
    # Carga del modelo de clasificación
    emotion_model = load_model('./model13.h5')
    # class_names = ["sadness", "happiness", "love", "anger", "worry", "neutral"]
    class_names = ["anger", "fear", "happy", "neutral", "sad", "surprise"]


    # Carga de la imagen
    image_path = "./inigo.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Error: no se pudo cargar la imagen.")
        return
    
    # Detectar y recortar el rostro
    face, confidence = detect_and_crop_face(image, yolo_model)
    if face is not None:
        print(f"Rostro detectado con confianza: {confidence*100:.2f}%")
        preprocessed_face = preprocess_image_from_array(face)

        # Predicción de emoción
        prediction = emotion_model.predict(preprocessed_face)
        print(prediction)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Mostrar la imagen con la predicción
        plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        plt.title(f"Emoción: {class_names[predicted_class]} ({confidence*100:.2f}%)")
        plt.axis("off")
        plt.show()
        return str(class_names[predicted_class])
    else:
        print("No se detectó ningún rostro en la imagen.")

if __name__ == "__main__":
    main()