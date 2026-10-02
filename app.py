import base64
import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Ancho de tolva de referencia en cm según modelo de camión (6860 mm = 686.0 cm)
ANCHO_TOLVA_CM = {
    "komatsu 830": 686.0,
    "cat 793": 890.0,
    "general": 686.0,
}


@app.route("/", methods=["GET"])
def index():
    return (
        jsonify(
            {
                "estado": "Servidor de Granulometría CAEX Activo",
                "version": "2.2",
            }
        ),
        200,
    )


@app.route("/analizar_granulometria", methods=["POST"])
def analizar_granulometria():
  print("\n=================== NUEVA PETICION RECIBIDA ===================")
  print(f"Content-Type: {request.content_type}")

  archivos = []

  try:
    # 1. Intentar capturar si viene en formato JSON con Base64 (Recomendado para Android moderno)
    if request.is_json:
      data = request.get_json()
      if "imagen_base64" in data:
        img_str = data["imagen_base64"]
        if "," in img_str:
          img_str = img_str.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        archivos.append(img_bytes)

    # 2. Intentar capturar si viene como multipart/form-data tradicional
    elif request.files:
      for key in request.files:
        for file_storage in request.files.getlist(key):
          archivos.append(file_storage.read())

    # 3. Respaldo por si llega como flujo binario directo en request.data
    elif request.data and len(request.data) > 0:
      archivos.append(request.data)

    if not archivos or len(archivos) == 0:
      print("Error: No se encontró ninguna imagen en la petición.")
      return (
          jsonify({"error": "No se recibió ninguna imagen para procesar."}),
          400,
      )

    camion = request.args.get("camion", "komatsu 830").lower()
    ancho_cm = ANCHO_TOLVA_CM.get(camion, 686.0)

    todos_diametros_cm = []
    resultados_individuales = []

    for idx, img_bytes in enumerate(archivos):
      # Decodificar bytes a imagen OpenCV
      np_arr = np.frombuffer(img_bytes, np.uint8)
      img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

      if img is None:
        continue

      alto_px, ancho_px = img.shape[:2]
      pixeles_por_cm = ancho_px / ancho_cm

      # Procesamiento de imagen para segmentación de clastos
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (7, 7), 0)
      _, thresh = cv2.threshold(
          blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
      )

      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

      contours, _ = cv2.findContours(
          opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )

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

      # Ordenar clastos por área (los más grandes al fondo)
      valid_contours = sorted(valid_contours, key=lambda x: x[1], reverse=True)

      img_clastos = img.copy()

      # Rellenar clastos con estilo WipFrag (colores según tamaño)
      for cnt, area in valid_contours:
        d_cm = (2 * np.sqrt(area / np.pi)) / pixeles_por_cm

        if d_cm > 80:
          color = (0, 128, 255)  # Naranjo para bloques grandes
        elif d_cm > 40:
          color = (0, 255, 0)  # Verde para tamaño mediano
        else:
          color = (0, 255, 255)  # Amarillo para clastos finos

        cv2.drawContours(img_clastos, [cnt], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(
            img_clastos, [cnt], -1, (50, 50, 50), thickness=1
        )  # Borde

      # --- DIBUJAR LÍNEA DE COTA LATERAL (6860 mm) ---
      x_cota = int(ancho_px * 0.88)
      y_inicio_cota = int(alto_px * 0.15)
      y_fin_cota = int(alto_px * 0.85)

      # Línea vertical principal roja
      cv2.line(
          img_clastos,
          (x_cota, y_inicio_cota),
          (x_cota, y_fin_cota),
          (0, 0, 255),
          3,
      )
      # Topes horizontales de la cota
      cv2.line(
          img_clastos,
          (x_cota - 12, y_inicio_cota),
          (x_cota + 12, y_inicio_cota),
          (0, 0, 255),
          3,
      )
      cv2.line(
          img_clastos,
          (x_cota - 12, y_fin_cota),
          (x_cota + 12, y_fin_cota),
          (0, 0, 255),
          3,
      )

      # Texto de medida exacta
      cv2.putText(
          img_clastos,
          "6860 mm",
          (x_cota - 95, int((y_inicio_cota + y_fin_cota) / 2)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 0, 255),
          2,
          cv2.LINE_AA,
      )
      # -----------------------------------------------

      # Codificar imagen procesada a JPG en base64
      _, buffer_clastos = cv2.imencode(".jpg", img_clastos)
      b64_clastos = base64.b64encode(buffer_clastos.getvalue()).decode("utf-8")

      resultados_individuales.append({
          "indice": idx + 1,
          "clastos_detectados": len(diametros_img),
          "imagen_clastos": b64_clastos,
      })

    # Cálculo de percentiles globales de la carga
    if len(todos_diametros_cm) > 0:
      arr_d = np.array(todos_diametros_cm)
      p99 = float(np.percentile(arr_d, 99))
      p80 = float(np.percentile(arr_d, 80))
      p50 = float(np.percentile(arr_d, 50))
      p20 = float(np.percentile(arr_d, 20))
    else:
      p99 = p80 = p50 = p20 = 0.0

    # Generación de gráfica combinada de distribución granulométrica (PNG)
    img_grafico = np.zeros((400, 600, 3), dtype=np.uint8)
    img_grafico.fill(255)
    cv2.putText(
        img_grafico,
        "Curva Granulometrica - CAEX",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        img_grafico,
        f"P99: {p99:.1f} cm | P80: {p80:.1f} cm",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
    )
    cv2.putText(
        img_grafico,
        f"P50: {p50:.1f} cm | P20: {p20:.1f} cm",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
    )
    cv2.putText(
        img_grafico,
        f"Tolva Calibrada: {ancho_cm} cm (6860 mm)",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # Dibujar ejes de la curva simulada
    cv2.line(img_grafico, (50, 320), (550, 320), (0, 0, 0), 2)
    cv2.line(img_grafico, (50, 220), (50, 320), (0, 0, 0), 2)
    if len(todos_diametros_cm) > 5:
      pts = np.array(
          [[50 + i * 5, 320 - int(min(val * 2, 90))] for i, val in enumerate(np.sort(arr_d)[:100]],
          np.int32,
      )
      if len(pts) > 1:
        cv2.polylines(img_grafico, [pts], False, (255, 0, 0), 3)

    _, buffer_grafico = cv2.imencode(".png", img_grafico)
    b64_grafico = base64.b64encode(buffer_grafico.getvalue()).decode("utf-8")

    response_data = {
        "p99_cm": f"{p99:.1f}",
        "p80_cm": f"{p80:.1f}",
        "p50_cm": f"{p50:.1f}",
        "p20_cm": f"{p20:.1f}",
        "grafico_combinado_base64": b64_grafico,
        "detalles_por_imagen": resultados_individuales,
    }

    print("Procesamiento exitoso. Retornando JSON al cliente.")
    return jsonify(response_data), 200

  except Exception as e:
    print(f"Excepción crítica en el servidor: {str(e)}")
    import traceback

    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
