import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Ancho de tolva de referencia en cm según modelo de camión
ANCHO_TOLVA_CM = {
    "komatsu 830": 850.0,
    "cat 793": 890.0,
    "general": 800.0
}

def generar_curva_wipfrag(diametros_cm):
    """Genera la gráfica acumulada tipo WipFrag e histograma y la devuelve en Base64."""
    diametros_in = np.array(diametros_cm) / 2.54
    diametros_sorted = np.sort(diametros_in)
    p_pasando = np.arange(1, len(diametros_sorted) + 1) / len(diametros_sorted) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(diametros_sorted, p_pasando, color='blue', linewidth=2, label='% Pasando')
    ax.set_xscale('log')
    ax.set_ylim(0, 105)
    ax.set_xlabel('Tamaño (in)', fontsize=10, fontweight='bold')
    ax.set_ylabel('% Pasando', fontsize=10, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    counts, bin_edges = np.histogram(diametros_sorted, bins=15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    freq_perc = (counts / len(diametros_sorted)) * 100
    ax.bar(bin_centers, freq_perc, width=np.diff(bin_edges), color='red', alpha=0.6, align='center')

    plt.title("Curva Granulométrica Combinada (Estilo WipFrag)", fontsize=12, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/analizar_granulometria', methods=['POST'])
def analizar_granulometria():
    print("\n=================== NUEVA PETICION JSON (BASE64) ===================")
    print(f"Content-Type: {request.content_type}")
    print(f"Parámetros GET (args): {request.args}")
    print("===================================================================\n")

    try:
        # Intentar leer los datos JSON enviados por la app
        data = request.get_json(silent=True)
        
        if not data or 'imagenes' not in data:
            return jsonify({"error": "No se encontró la clave 'imagenes' en formato JSON."}), 400

        lista_imagenes_b64 = data['imagenes']
        
        if not lista_imagenes_b64 or len(lista_imagenes_b64) == 0:
            return jsonify({"error": "La lista de imágenes está vacía."}), 400

        # Obtener modelo de camión para la escala física
        camion = request.args.get('camion', 'komatsu 830').lower()
        ancho_cm = ANCHO_TOLVA_CM.get(camion, 850.0)

        todos_diametros_cm = []
        resultados_individuales = []

        for idx, img_b64 in enumerate(lista_imagenes_b64):
            try:
                # Limpiar la cabecera data:image si la app la incluye por error
                if ',' in img_b64:
                    img_b64 = img_b64.split(',')[1]

                # Decodificar Base64 a bytes y luego a matriz OpenCV
                imagen_bytes = base64.b64decode(img_b64)
                nparr = np.frombuffer(imagen_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                # Escala física para esta imagen
                ancho_px = img.shape[1]
                pixeles_por_cm = ancho_px / ancho_cm

                # Procesamiento OpenCV
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
                    if area > 10:  # Filtrar ruido
                        diametro_px = 2 * np.sqrt(area / np.pi)
                        d_cm = diametro_px / pixeles_por_cm
                        diametros_img.append(d_cm)
                        todos_diametros_cm.append(d_cm)
                        valid_contours.append(cnt)

                # Dibujar contornos individuales en verde
                img_clastos = img.copy()
                cv2.drawContours(img_clastos, valid_contours, -1, (0, 255, 0), 2)

                _, buffer_clastos = cv2.imencode('.jpg', img_clastos)
                b64_clastos = base64.b64encode(buffer_clastos.getvalue()).decode('utf-8')

                resultados_individuales.append({
                    "indice": idx + 1,
                    "clastos_detectados": len(diametros_img),
                    "imagen_clastos": b64_clastos
                })

            except Exception as ex_img:
                print(f"Error procesando imagen índice {idx}: {str(ex_img)}")
                continue

        if not todos_diametros_cm:
            return jsonify({"error": "No se detectaron clastos válidos en ninguna de las imágenes enviadas."}), 400

        # --- CÁLCULO DE PERCENTILES GLOBALES ---
        todos_diametros_cm.sort()
        p99_cm = float(np.percentile(todos_diametros_cm, 99))
        p80_cm = float(np.percentile(todos_diametros_cm, 80))
        p50_cm = float(np.percentile(todos_diametros_cm, 50))
        p20_cm = float(np.percentile(todos_diametros_cm, 20))

        # Generar la curva WipFrag combinada
        grafico_combinado_b64 = generar_curva_wipfrag(todos_diametros_cm)

        print(f"--> PROCESAMIENTO EXITOSO ({len(resultados_individuales)} fotos procesadas)")
        return jsonify({
            "total_fotos": len(resultados_individuales),
            "total_clastos_analizados": len(todos_diametros_cm),
            "p99_cm": round(p99_cm, 2),
            "p80_cm": round(p80_cm, 2),
            "p50_cm": round(p50_cm, 2),
            "p20_cm": round(p20_cm, 2),
            "p99_in": round(p99_cm / 2.54, 2),
            "p80_in": round(p80_cm / 2.54, 2),
            "p50_in": round(p50_cm / 2.54, 2),
            "p20_in": round(p20_cm / 2.54, 2),
            "grafico_combinado_base64": grafico_combinado_b64,
            "detalles_por_imagen": resultados_individuales
        }), 200

    except Exception as e:
        print(f"EXCEPCION EN EL SERVIDOR: {str(e)}")
        return jsonify({"error": f"Excepción en servidor: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
