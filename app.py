import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analizar_granulometria', methods=['POST'])
def analizar_granulometria():
    image_bytes = None

    # 1. Intenta obtener la imagen por formulario Multipart ('file' o 'imagen')
    if 'file' in request.files:
        image_bytes = request.files['file'].read()
    elif 'imagen' in request.files:
        image_bytes = request.files['imagen'].read()
    # 2. Si viene por Web1.PostFile de App Inventor (envío de datos binarios directos)
    elif request.data:
        image_bytes = request.data

    # Validar si se recibieron bytes de imagen
    if not image_bytes:
        return jsonify({"error": "No se adjunto ninguna imagen"}), 400

    # Convertir bytes a matriz OpenCV (numpy)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400

    # ----------------------------------------------------
    # AQUÍ VA TU LÓGICA DE PROCESAMIENTO OPENCV
    # (Ajusta los cálculos según tu modelo/escala)
    # ----------------------------------------------------
    
    # Valores de prueba / ejemplo del cálculo granulométrico en cm:
    p99 = 45.2
    p80 = 28.5
    p50 = 15.1
    p20 = 6.4

    # Respuesta JSON estructurada exactamente con las claves que espera App Inventor
    return jsonify({
        "p99_cm": round(p99, 2),
        "p80_cm": round(p80, 2),
        "p50_cm": round(p50, 2),
        "p20_cm": round(p20, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
