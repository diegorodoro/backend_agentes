from flask import Flask, request, jsonify
import base64
from utils.chatgpt_config import send_ChatGPT


app = Flask(__name__)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        base64_image = data.get('base64_image')

        if not base64_image:
            return jsonify({"error": "Missing 'base64_image' in the request"}), 400

        image_data = base64.b64decode(base64_image) 
        with open("uploaded_image.jpeg", "wb") as img_file:
            img_file.write(image_data)

        return jsonify({"message": "Image received and saved successfully!",
                        "chat":send_ChatGPT("sad")}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/upload_text', methods=['POST'])
def upload_text():
    try:
        # Obtener el cuerpo de la solicitud como JSON
        data = request.get_json()
        
        # Verificar que el campo 'text' esté presente y sea una cadena
        text = data.get('text')
        
        if text is None:
            return jsonify({"error": "Falta el campo 'text' en la solicitud."}), 400
        
        # Verificar que 'text' sea una cadena de texto
        if not isinstance(text, str):
            return jsonify({"error": "El campo 'text' debe ser una cadena de texto."}), 400
        
        # Verificar que 'text' no sea vacío o solo espacios
        if not text.strip():
            return jsonify({"error": "El campo 'text' no debe estar vacío."}), 400

        # Procesar el texto (reemplaza con tu lógica)
        return jsonify({
            "message": "¡Texto recibido y guardado exitosamente!",
            "chat": send_ChatGPT("sad")  # Reemplaza con tu función real
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
