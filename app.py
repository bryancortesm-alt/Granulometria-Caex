import os
import base64
import io
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Ancho de tolva de referencia en cm (6860 mm = 686.0 cm)
ANCHO_TOLVA_CM = {
    "komatsu 830": 686.0,
    "cat 793": 890.0,
    "general": 686.0
}

@app.route('/analizar_granulometria', methods=['POST'])
def analizar_granulometria():
    print("\n=================== NUEVA PETICION RECIBIDA ===================")
    print(f"Content-Type: {request.content_type}")
    print(f"Archivos en request.files: {list(request.files.keys())}")
    print(f"Tamaño de request.data: {len(request.data) if request.data else 0} bytes")
    print("===============================================================\n")

    archivos = []

    # 1. Capturar si viene por multipart/form-data
    if request.files:
        for key in request.files:
            archivos.extend(request.files.getlist(key))

    # 2. Si request.files está vacío, capturar el flujo binario directo
    if not archivos and request.data and len(request.data) > 0:
        archivos = [request.data]

    if not archivos or len(archivos) == 0:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    camion = request.args.get('camion', 'komatsu 830').lower()
    malla = request.args.get('malla', 'malla_1').lower()
    ancho_cm = ANCHO_TOLVA_CM.get(camion, 686.0)

    todos_diametros_cm = []
    resultados_individuales = []

    for idx, archivo in enumerate(archivos):
        if hasattr(archivo, 'read'):
            file_bytes = np.frombuffer(archivo.read(), np.uint8)
        else:
            file_bytes = np.frombuffer(archivo, np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        ancho_px = img.shape[1]
        alto_px = img.shape[0]
        pixeles_por_cm = ancho_px / ancho_cm

        # Procesamiento de imagen con OpenCV
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

        # Ordenar clastos por área (los más grandes primero)
        valid_contours = sorted(valid_contours, key=lambda x: x[1], reverse=True)

        img_clastos = img.copy()

        # Rellenar clastos con colores simulando el estándar WipFrag
        for cnt, area in valid_contours:
            d_cm = (2 * np.sqrt(area / np.pi)) / pixeles_por_cm
            
            if d_cm > 80:
                color = (0, 128, 255)   # Naranjo / Rojo para bloques muy grandes
            elif d_cm > 40:
                color = (0, 255, 0)     # Verde para tamaño mediano
            else:
                color = (0, 255, 255)   # Amarillo para clastos finos

            cv2.drawContours(img_clastos, [cnt], -1, color, thickness=cv2.FILLED)
            cv2.drawContours(img_clastos, [cnt], -1, (50, 50, 50), thickness=1)

        # --- LÍNEA DE COTA LATERAL (6860 mm) ---
        x_cota = int(ancho_px * 0.85)
        y_inicio_cota = int(alto_px * 0.2)
        y_fin_cota = int(alto_px * 0.85)

        cv2.line(img_clastos, (x_cota, y_inicio_cota), (x_cota, y_fin_cota), (0, 0, 255), 3)
        cv2.line(img_clastos, (x_cota - 10, y_inicio_cota), (x_cota + 10, y_inicio_cota), (0, 0, 255), 3)
        cv2.line(img_clastos, (x_cota - 10, y_fin_cota), (x_cota + 10, y_fin_cota), (0, 0, 255), 3)

        cv2.putText(img_clastos, "6860 mm", (x_cota - 95, int((y_inicio_cota + y_fin_cota) / 2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # --------------------------------------

        _, buffer_clastos = cv2.imencode('.jpg', img_clastos)
        b64_clastos = base64.b64encode(buffer_clastos.getvalue()).decode('utf-8')

        resultados_individuales.append({
            "indice": idx + 1,
            "clastos_detectados": len(diametros_img),
            "imagen_clastos": b64_clastos
        })

    if not todos_diametros_cm:
        return jsonify({"error": "No se pudieron detectar clastos en las imágenes."}), 400

    # Cálculo de percentiles globales
    todos_diametros_cm.sort()
    arr = np.array(todos_diametros_cm)
    p99 = float(np.percentile(arr, 99))
    p80 = float(np.percentile(arr, 80))
    p50 = float(np.percentile(arr, 50))
    p20 = float(np.percentile(arr, 20))

    # Generación de la curva granulométrica global combinada
    plt.figure(figsize=(6, 4))
    counts, bin_edges = np.histogram(arr, bins=20, density=True)
    cdf = np.cumsum(counts * np.diff(bin_edges)) * 100
    
    plt.plot(bin_edges[:-1], cdf, color='blue', linewidth=2, label=f'Curva {malla.upper()}')
    plt.title(f'Curva Granulométrica - {malla.upper()}')
    plt.xlabel('Tamaño (cm)')
    plt.ylabel('% Acumulado')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    grafico_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({
        "malla": malla,
        "p99_cm": round(p99, 2),
        "p80_cm": round(p80, 2),
        "p50_cm": round(p50, 2),
        "p20_cm": round(p20, 2),
        "grafico_combinado_base64": grafico_b64,
        "detalles_por_imagen": resultados_individuales
    })


@app.route('/generar_informe_pdf', methods=['POST'])
def generar_informe_pdf():
    data = request.json or {}
    p99 = data.get('p99_cm', 0)
    p80 = data.get('p80_cm', 0)
    p50 = data.get('p50_cm', 0)
    p20 = data.get('p20_cm', 0)
    camion = data.get('camion', 'Komatsu 830')
    malla = data.get('malla', 'Malla 1')

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Encabezado
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "INFORME DE GRANULOMETRÍA - CAEX")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, 730, f"Sector / Malla: {malla.upper()}")
    c.drawString(50, 715, f"Modelo de Camión: {camion}")
    c.drawString(50, 700, f"Referencia de Tolva: 6860 mm (686.0 cm)")
    
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, "Resultados de Percentiles:")
    
    c.setFont("Helvetica", 11)
    c.drawString(70, 635, f"• P99: {p99} cm")
    c.drawString(70, 615, f"• P80: {p80} cm")
    c.drawString(70, 595, f"• P50: {p50} cm")
    c.drawString(70, 575, f"• P20: {p20} cm")
    
    # Pie de página
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Generado automáticamente por el Sistema de Análisis Granulométrico Móvil.")

    c.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name=f'Informe_{malla}.pdf')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
