from fastapi import FastAPI, HTTPException
import os
import cv2
# import ollama
from ultralytics import YOLO
# import numpy as np
from google import genai
from google.genai import types

import PIL.Image

app = FastAPI()

model = YOLO('best.pt')  # Modelo YOLO para detección de placas

@app.get('/')
def status():
    return {"status": "running"}

@app.post('/detection_plate')
async def search_plate(image_path: str, confidence: float = 0.5):
    

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"La imagen no existe en la ruta: {image_path}")

    try:

        client = genai.Client(api_key="${GEMINI_API_KEY}")
        # Leer la imagen
        image = cv2.imread(image_path)
        
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen. Verifique que el formato sea válido.")

        # Detección de placas con YOLO
        results = model.predict(source=image_path, conf=confidence)
        
        if len(results[0].boxes) == 0:
            return {"error": "No se detectó ninguna placa"}

        detected_plates = []
        # print('result::::::::::::::::::::::: ',results[0].boxes)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Recortar la placa detectada
            plate_crop = image[y1:y2, x1:x2]

            # Convertir a escala de grises y aplicar preprocesamiento
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 1)
            _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Guardar la imagen procesada temporalmente
            cv2.imwrite("temp_plate.jpg", plate_gray)
            # Leer la imagen
            image = PIL.Image.open("temp_plate.jpg")
            # model Gemini 2 Vision
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Extrae solo los caracteres alfanuméricos de la matrícula en esta imagen", image])
            
            print(response.text)

            plate_text = response.text

            detected_plates.append({
                "box": [x1, y1, x2, y2],
                "plate_text": plate_text
            })

        return {"detected_plates": detected_plates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento de la imagen: {str(e)}")
