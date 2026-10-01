import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

ANCHO_TOLVA_CM = {
    "komatsu 830": 850.0,
    "cat 793": 890.0,
    "general": 800.0
}

def generar_curva_wipfrag(diametros_cm):
    """Genera la gráfica acumulada tipo WipFrag y la devuelve en Base64."""
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

    plt.title("Curva Granulométrica (Estilo WipFrag)", fontsize=12, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/analizar_granulometria', methods=['POST'])
def analizar_granulometria():
    print("\n=================== NUEVA PETICION RECIBIDA ===================")
    print(f"Content-Type: {request.content_type}")
    print(f"Archivos en request.files: {list(request.files.keys())}")
    print(f"Tamaño de request.data (bytes): {len(request.data) if request.data else 0}")
    print(f"Parámetros GET (args): {request.args}")
    print("===============================================================\n")

    imagen_bytes = None

    # 1. Buscar en multipart/form-data (request.files)
    if request.files:
        primer_key = list(request.files.keys())[0]
        print(f"--> Imagen detectada en request.files['{primer_key}']")
        imagen_bytes = request.files[primer_key].read()

    # 2. Buscar en datos binarios directos (Web1.PostFile de App Inventor)
    elif request.data and len(request.data) > 0:
        print(f"--> Imagen detectada en request.data ({len(request.data)} bytes)")
        imagen_bytes = request.data

    # Si no se detectaron bytes de imagen
    if not imagen_bytes or len(imagen_bytes) == 0:
        mensaje_error = f"No se recibio ninguna imagen. Content-Type: {request.content_type}, Data Size: {len(request.data) if request.data else 0}"
        print(f"ERROR: {mensaje_error}")
        return jsonify({"error": mensaje_error}), 400

    try:
        # Decodificar la imagen a formato OpenCV
        nparr = np.frombuffer(imagen_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("ERROR: OpenCV no pudo decodificar la imagen.")
            return jsonify({"error": "No se pudo decodificar la imagen (formato no valido o corrupto)"}), 400

        # Obtener modelo de camión desde el parámetro URL (por defecto komatsu 830)
        camion = request.args.get('camion', 'komatsu 830').lower()
        ancho_cm = ANCHO_TOLVA_CM.get(camion, 850.0)

        # Procesamiento con OpenCV
        ancho_px = img.shape[1]
        pixeles_por_cm = ancho_px / ancho_cm

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diametros_cm = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:  # Filtrar ruido
                diametro_px = 2 * np.sqrt(area / np.pi)
                diametros_cm.append(diametro_px / pixeles_por_cm)

        if not diametros_cm:
            print("ERROR: No se detectaron clastos en la imagen.")
            return jsonify({"error": "No se detectaron clastos en la imagen"}), 400

        # Cálculo de percentiles
        diametros_cm.sort()
        p99_cm = float(np.percentile(diametros_cm, 99))
        p80_cm = float(np.percentile(diametros_cm, 80))
        p50_cm = float(np.percentile(diametros_cm, 50))
        p20_cm = float(np.percentile(diametros_cm, 20))

        # Generar gráfica Base64
        grafico_b64 = generar_curva_wipfrag(diametros_cm)

        print("--> PROCESAMIENTO EXITOSO")
        return jsonify({
            "p99_cm": round(p99_cm, 2),
            "p80_cm": round(p80_cm, 2),
            "p50_cm": round(p50_cm, 2),
            "p20_cm": round(p20_cm, 2),
            "p99_in": round(p99_cm / 2.54, 2),
            "p80_in": round(p80_cm / 2.54, 2),
            "p50_in": round(p50_cm / 2.54, 2),
            "p20_in": round(p20_cm / 2.54, 2),
            "grafico_base64": grafico_b64
        }), 200

    except Exception as e:
        print(f"EXCEPCION EN EL SERVIDOR: {str(e)}")
        return jsonify({"error": f"Excepción en servidor: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
