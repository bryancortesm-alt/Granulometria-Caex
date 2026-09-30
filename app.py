import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite conexion desde la app movil

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Servidor de Analisis Granulometrico Activo"
    })

@app.route('/analizar_granulometria', methods=['POST'])
def analizar():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se adjunto ninguna imagen'}), 400
        
        file = request.files['image']
        ancho_tolva_cm = float(request.form.get('ancho_tolva_cm', 750)) 

        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'La imagen no pudo ser procesada'}), 400

        _, ancho_px, _ = img.shape
        px_por_cm = ancho_px / ancho_tolva_cm

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diametros_cm = []
        for c in contours:
            area_px = cv2.contourArea(c)
            if area_px > 25:
                diametro_px = 2 * np.sqrt(area_px / np.pi)
                diametros_cm.append(diametro_px / px_por_cm)

        if len(diametros_cm) < 3:
            return jsonify({'error': 'No se detectaron suficientes clastos en la tolva. Intente con una foto con mejor iluminacion.'}), 400

        diametros_cm = np.sort(diametros_cm)
        pasante = np.linspace(0, 100, len(diametros_cm))
        
        p80 = float(np.percentile(diametros_cm, 80))
        p50 = float(np.percentile(diametros_cm, 50))
        p20 = float(np.percentile(diametros_cm, 20))

        pasos = max(1, len(diametros_cm) // 50)
        diametros_resumidos = [round(d, 2) for d in diametros_cm[::pasos].tolist()]
        pasante_resumido = [round(p, 2) for p in pasante[::pasos].tolist()]

        return jsonify({
            'exito': True,
            'p80_cm': round(p80, 2),
            'p50_cm': round(p50, 2),
            'p20_cm': round(p20, 2),
            'total_clastos': len(diametros_cm),
            'curva': {
                'diametros_cm': diametros_resumidos,
                'pasante_pct': pasante_resumido
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
