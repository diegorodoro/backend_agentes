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

from openai import OpenAI
client = OpenAI()

def send_ChatGPT(emotion):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"what do you recommend if im feeling {emotion}?\n"
          }
        ]
      }
    ],
    response_format={
      "type": "text"
    },
    temperature=0.81,
    max_tokens=2000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].message.content