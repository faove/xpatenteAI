from fastapi import FastAPI, HTTPException, File, UploadFile
import os
from models.DetectionPlate import DetectionPlate
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
import numpy as np
from google import genai
from google.genai import types

import PIL.Image

app = FastAPI()

model = YOLO('best.pt')  # Modelo YOLO para detección de placas

@app.get('/')
def status():
    return {"status": "running"}

@app.post('/detection_plate')
async def search_plate(file: UploadFile = File(...), confidence: float = 0.5):
    try:
        # Leer la imagen en memoria
        contents = await file.read()
        image_np = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Cargar las variables desde .env
        load_dotenv()

        # Obtener la clave de la variable de entorno
        api_key = os.getenv("GEMINI_API_KEY")
        # print('api_key::::::::::::::::::::::: ',api_key)
        # return {"api_key": api_key}
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en .env")
        
        # Leer la imagen
        # image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen. Verifique que el formato sea válido.")

        # Detección de placas con YOLO
        results = model.predict(source=image, conf=confidence)
        
        if len(results[0].boxes) == 0:
            return {"error": "No se detectó ninguna placa"}

        detected_plates = []
        
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
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Extrae solo los caracteres alfanuméricos de la matrícula en esta imagen", image])
            

            plate_text = response.text
            detected_plates.append({"plate": plate_text})

        return {"results": detected_plates}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento de la imagen: {str(e)}")
