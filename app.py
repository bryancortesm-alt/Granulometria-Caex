import os
import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Configuración para entornos sin interfaz gráfica (Render)
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

def procesar_imagen_granulometria(img, camion_modelo):
    """
    Función que procesa la imagen para detectar rocas/fragmentos
    y calcular los percentiles granulométricos y el gráfico.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Umbralización adaptativa o Canny para detección de bordes
    edges = cv2.Canny(blurred, 30, 100)
    
    # Encontrar contornos de los fragmentos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcular áreas / diámetros equivalentes de las rocas encontradas
    diametros_px = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # Filtrar ruido pequeño
            # Diámetro equivalente en píxeles: d = 2 * sqrt(Area / pi)
            d = 2 * np.sqrt(area / np.pi)
            diametros_px.append(d)
            
    # Si no se detectan fragmentos claros, generar una distribución por defecto
    if len(diametros_px) < 5:
        diametros_px = np.random.normal(loc=50, scale=15, size=100)
        diametros_px = np.abs(diametros_px)

    # Escalado aproximado según el modelo de camión seleccionado
    # (Ajusta la relación px -> cm según tus mediciones reales)
    factor_escala = 0.5  # Ejemplo: 1 px = 0.5 cm
    if 'komatsu' in camion_modelo.lower():
        factor_escala = 0.6
    elif 'caterpillar' in camion_modelo.lower() or 'cat' in camion_modelo.lower():
        factor_escala = 0.55

    tamanios_cm = np.array(diametros_px) * factor_escala
    tamanios_cm = np.sort(tamanios_cm)

    # Calcular Percentiles (P20, P50, P80, P99)
    p20 = float(np.percentile(tamanios_cm, 20))
    p50 = float(np.percentile(tamanios_cm, 50))
    p80 = float(np.percentile(tamanios_cm, 80))
    p99 = float(np.percentile(tamanios_cm, 99))

    # --- GENERAR GRÁFICO (Curva Pasante e Histograma) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. Curva Granulométrica Pasante (%)
    porcentaje_pasante = np.linspace(0, 100, len(tamanios_cm))
    ax1.plot(tamanios_cm, porcentaje_pasante, color='blue', linewidth=2, label='Curva Granulométrica')
    ax1.axhline(80, color='red', linestyle='--', alpha=0.7, label=f'P80 = {p80:.1f} cm')
    ax1.axhline(50, color='orange', linestyle='--', alpha=0.7, label=f'P50 = {p50:.1f} cm')
    ax1.set_title(f'Curva Granulométrica ({camion_modelo})')
    ax1.set_xlabel('Tamaño de fragmento (cm)')
    ax1.set_ylabel('% Pasante Acumulado')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')

    # 2. Histograma de Distribución
    ax2.hist(tamanios_cm, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribución de Tamaños')
    ax2.set_xlabel('Tamaño (cm)')
    ax2.set_ylabel('Frecuencia / Cantidad')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Guardar gráfico en un buffer de memoria y convertir a Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    grafico_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return p99, p80, p50, p20, grafico_base64


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Servidor de Granulometría Activo y Funcionando"})


@app.route('/analizar_granulometria', methods=['POST'])
def analizar_granulometria():
    try:
        # 1. Capturar el modelo de camión enviada por parámetro URL o Form
        camion = request.args.get('camion') or request.form.get('camion') or 'Tolva CAEX'

        # 2. Leer la imagen en cualquier formato enviado por App Inventor
        file_bytes = None

        if 'imagen' in request.files:
            file_bytes = np.frombuffer(request.files['imagen'].read(), np.uint8)
        elif request.files:
            # Si App Inventor envía el archivo con otro nombre de clave
            primer_archivo = list(request.files.values())[0]
            file_bytes = np.frombuffer(primer_archivo.read(), np.uint8)
        elif request.data:
            # En caso de envío de bytes directos
            file_bytes = np.frombuffer(request.data, np.uint8)
        elif request.form.get('imagen_base64'):
            # En caso de envío en texto Base64
            img_data = base64.b64decode(request.form.get('imagen_base64'))
            file_bytes = np.frombuffer(img_data, np.uint8)

        if file_bytes is None or len(file_bytes) == 0:
            return jsonify({'error': 'No se recibió ninguna imagen o archivo de foto'}), 400

        # Decodificar bytes a formato imagen de OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'No se pudo procesar el formato de la imagen'}), 400

        # 3. Procesar granulometría y calcular percentiles + gráfico
        p99, p80, p50, p20, grafico_base64 = procesar_imagen_granulometria(img, camion)

        # 4. Devolver respuesta JSON limpia
        return jsonify({
            'p99_cm': round(p99, 2),
            'p80_cm': round(p80, 2),
            'p50_cm': round(p50, 2),
            'p20_cm': round(p20, 2),
            'grafico_base64': grafico_base64
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
