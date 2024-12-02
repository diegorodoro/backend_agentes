from openai import OpenAI
from dotenv import load_dotenv
import os

# Ruta relativa al archivo .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Cargar el archivo .env desde la ruta especificada
load_dotenv(dotenv_path)

# Leer la clave API
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

def send_ChatGPT(emotion_face=None, emotion_text=None):
    # Verificar qué parámetros se proporcionaron
    if emotion_face and emotion_text:
        prompt = f"Act like you are a psychologist expert in emotions, i want recommendation of music, fun activities, actions, or advices if the image im receiving shows a {emotion_face} emotion and the text from a chat that the user provide us shows a {emotion_text} emotion"
    elif emotion_face:
        prompt = f"i want recommendation of music, fun activities, actions, or advices because im expiriencing a {emotion_face} emotion"
    elif emotion_text:
        prompt = f"i want recommendation of music, fun activities, actions, or advices because im expiriencing a {emotion_text} emotion"
    else:
        raise ValueError("At least one of 'emotion_face' or 'emotion_text' must be provided.")
    
    # Realizar la llamada al modelo
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.81,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content
