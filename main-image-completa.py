from fastapi import FastAPI, HTTPException
import os
import cv2
import ollama
from ultralytics import YOLO
import numpy as np

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
        # Leer la imagen
        image = cv2.imread(image_path)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen. Verifique que el formato sea válido.")
        

        # Guardar la imagen procesada temporalmente
        cv2.imwrite("temp_plate.jpg", image)
        

        # Llama 3.2 Vision para OCR
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': 'Extrae solo los caracteres alfanuméricos de la matrícula en esta imagen',
                'images': ['/var/www/html/xpatenteAI/temp_plate.jpg']
            }]
        )

        plate_text = response['message']['content'].strip()

        detected_plates.append({
            "box": [x1, y1, x2, y2],
            "plate_text": plate_text
        })

        return {"detected_plates": detected_plates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento de la imagen: {str(e)}")
