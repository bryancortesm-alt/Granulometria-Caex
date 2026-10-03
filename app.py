import os
import base64
import cv2
import numpy as np
from urllib.parse import parse_qs
from flask import Flask, request, jsonify

app = Flask(__name__)

ANCHO_TOLVA_CM = {
    "komatsu 830": 686.0,
    "cat 793": 890.0,
    "general": 686.0,
}

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "estado": "Servidor de Granulometria CAEX Activo",
        "version": "2.8"
    }), 200

@app.route("/analizar_granulometria", methods=["POST"])
def analizar_granulometria():
    print("\n=================== NUEVA PETICION RECIBIDA ===================")
    print(f"Content-Type: {request.content_type}")
    
    archivos = []
    
    try:
        # 1. Revisar request.form tradicional
        if request.form:
            for key in request.form:
                val = request.form[key]
                if val and len(val) > 50:
                    try:
                        limpio = val.split(",")[1] if "," in val else val
                        decodificado = base64.b64decode(limpio)
                        if len(decodificado) > 50:
                            archivos.append(decodificado)
                    except Exception:
                        pass

        # 2. Revisar request.data (por si viene como urlencoded en el cuerpo crudo o texto plano)
        if not archivos and request.data:
            cuerpo_bytes = request.data
            try:
                cuerpo_str = cuerpo_bytes.decode('utf-8', errors='ignore').strip()
                # Si viene en formato key=value o key=data:image...
                if "=" in cuerpo_str:
                    parsed = parse_qs(cuerpo_str)
                    for k, vals in parsed.items():
                        for val in vals:
                            if len(val) > 50:
                                limpio = val.split(",")[1] if "," in val else val
                                decodificado = base64.b64decode(limpio)
                                if len(decodificado) > 50:
                                    archivos.append(decodificado)
                                    print(f"Imagen extraída desde parámetro URL-encoded: {k}")
                else:
                    # Intentar base64 directo
                    limpio = cuerpo_str.split(",")[1] if "," in cuerpo_str else cuerpo_str
                    archivos.append(base64.b64decode(limpio))
            except Exception as e:
                print(f"Error procesando request.data: {e}")

        # 3. Revisar archivos multipart tradicionales
        if not archivos and request.files:
            for key in request.files:
                for file_storage in request.files.getlist(key):
                    archivos.append(file_storage.read())

        if not archivos or len(archivos) == 0:
            print("Error crítico: No se pudo extraer ninguna imagen válida.")
            return jsonify({"error": "No se encontró ninguna imagen en la petición."}), 400

        camion = request.args.get('camion', 'komatsu 830').lower()
        ancho_cm = ANCHO_TOLVA_CM.get(camion, 686.0)
        
        todos_diametros_cm = []
        resultados_individuales = []

        for idx, img_bytes in enumerate(archivos):
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            alto_px, ancho_px = img.shape[:2]
            pixeles_por_cm = ancho_px / ancho_cm

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            diametros_img = []
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10:
                    diametro_px = 2 * np.sqrt(area / np.pi)
                    d_cm = diametro_px / pixeles_por_cm
                    diametros_img.append(d_cm)
                    todos_diametros_cm.append(d_cm)
                    valid_contours.append((cnt, area))

            valid_contours = sorted(valid_contours, key=lambda x: x[1], reverse=True)
            img_clastos = img.copy()

            for cnt, area in valid_contours:
                d_cm = (2 * np.sqrt(area / np.pi)) / pixeles_por_cm
                if d_cm > 80:
                    color = (0, 128, 255)   # Naranjo
                elif d_cm > 40:
                    color = (0, 255, 0)     # Verde
                else:
                    color = (0, 255, 255)   # Amarillo

                cv2.drawContours(img_clastos, [cnt], -1, color, thickness=cv2.FILLED)
                cv2.drawContours(img_clastos, [cnt], -1, (50, 50, 50), thickness=1)

            # Cota lateral 6860 mm
            x_cota = int(ancho_px * 0.88)
            y_inicio_cota = int(alto_px * 0.15)
            y_fin_cota = int(alto_px * 0.85)

            cv2.line(img_clastos, (x_cota, y_inicio_cota), (x_cota, y_fin_cota), (0, 0, 255), 3)
            cv2.line(img_clastos, (x_cota - 12, y_inicio_cota), (x_cota + 12, y_inicio_cota), (0, 0, 255), 3)
            cv2.line(img_clastos, (x_cota - 12, y_fin_cota), (x_cota + 12, y_fin_cota), (0, 0, 255), 3)
            cv2.putText(img_clastos, "6860 mm", (x_cota - 95, int((y_inicio_cota + y_fin_cota) / 2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            _, buffer_clastos = cv2.imencode('.jpg', img_clastos)
            b64_clastos = base64.b64encode(buffer_clastos.getvalue()).decode('utf-8')

            resultados_individuales.append({
                "indice": idx + 1,
                "clastos_detectados": len(diametros_img),
                "imagen_clastos": b64_clastos
            })

        if len(todos_diametros_cm) > 0:
            arr_d = np.array(todos_diametros_cm)
            p99 = float(np.percentile(arr_d, 99))
            p80 = float(np.percentile(arr_d, 80))
            p50 = float(np.percentile(arr_d, 50))
            p20 = float(np.percentile(arr_d, 20))
        else:
            p99 = p80 = p50 = p20 = 0.0

        img_grafico = np.zeros((400, 600, 3), dtype=np.uint8)
        img_grafico.fill(255)
        cv2.putText(img_grafico, "Curva Granulometrica - CAEX", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img_grafico, f"P99: {p99:.1f} cm | P80: {p80:.1f} cm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        cv2.putText(img_grafico, f"P50: {p50:.1f} cm | P20: {p20:.1f} cm", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        cv2.putText(img_grafico, f"Tolva Calibrada: {ancho_cm} cm (6860 mm)", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.line(img_grafico, (50, 320), (550, 320), (0, 0, 0), 2)
        cv2.line(img_grafico, (50, 220), (50, 320), (0, 0, 0), 2)
        
        if len(todos_diametros_cm) > 5:
            pts = np.array([[50 + i * 5, 320 - int(min(val * 2, 90))] for i, val in enumerate(np.sort(arr_d)[:100])], np.int32)
            if len(pts) > 1:
                cv2.polylines(img_grafico, [pts], False, (255, 0, 0), 3)

        _, buffer_grafico = cv2.imencode('.png', img_grafico)
        b64_grafico = base64.b64encode(buffer_grafico.getvalue()).decode('utf-8')

        return jsonify({
            "p99_cm": f"{p99:.1f}",
            "p80_cm": f"{p80:.1f}",
            "p50_cm": f"{p50:.1f}",
            "p20_cm": f"{p20:.1f}",
            "grafico_combinado_base64": b64_grafico,
            "detalles_por_imagen": resultados_individuales
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
