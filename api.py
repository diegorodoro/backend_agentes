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
                        "chat":send_ChatGPT(emotion_face="sad")}), 200

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
            "chat": send_ChatGPT(emotion_text="sad")  # Reemplaza con tu función real
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/upload_info', methods=['POST'])
def upload_info():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be in JSON format."}), 400

        base64_image = data.get('img')
        text = data.get('text')

        print(base64_image,text)

        # Validar presencia de los campos requeridos
        if not base64_image:
            return jsonify({"error": "Missing 'img' in the request."}), 400
        
        if text is None:
            return jsonify({"error": "Missing 'text' in the request."}), 400
        
        # Validar que 'text' sea una cadena no vacía
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "'text' must be a non-empty string."}), 400

        # Decodificar la imagen y guardarla
        try:
            image_data = base64.b64decode(base64_image)
            with open("uploaded_image.jpeg", "wb") as img_file:
                img_file.write(image_data)

        except (base64.binascii.Error, TypeError) as decode_error:
            return jsonify({"error": "Invalid base64 string for 'img'."}), 400

        # Simular la función send_ChatGPT
        chat_response = send_ChatGPT("sad","happy")  # Sustituir por la implementación real

        return jsonify({
            "message": "Image received and saved successfully!",
            "chat": chat_response
        }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
