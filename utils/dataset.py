import pandas as pd
from sklearn.utils import resample
import re
import os

file_path=os.path.join(os.path.dirname(__file__),'emotions.csv')

data=pd.read_csv(file_path)

# Ver distribución de clases después de la concatenación
class_counts = data["label"].value_counts()
print("Distribución después de la concatenación:")
print(class_counts)

min_class_count = class_counts.min()

# Balancear el dataset
balanced_data = pd.concat([
    resample(data[data["label"] == label], 
             replace=False,  # No hacer duplicados
             n_samples=min_class_count,  # Igualar al tamaño de la clase minoritaria
             random_state=42) 
    for label in class_counts.index
])

balanced_data = balanced_data.sample(frac=1, random_state=42)

def clean_text(text):
    # Paso 1: Remover links completos que contengan http, https, o www
    text = re.sub(r"(http[s]?://[^\s]+|www\.[^\s]+)", "", text)
    # Paso 2: Remover caracteres especiales y dejar solo letras, números y espacios
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    # Paso 3: Eliminar espacios extra
    text = re.sub(r"\s+", " ", text).strip()
    return text

balanced_data["text"] = balanced_data["text"].apply(clean_text)

# Mostrar resultados finales
print("\nDistribución balanceada de clases:")
print(balanced_data["label"].value_counts())

# Guardar el dataset balanceado
balanced_data.to_csv("balanced_dataset.csv", index=False)