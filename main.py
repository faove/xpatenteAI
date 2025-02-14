from fastapi import FastAPI, HTTPException
import os
from models.DetectionPlate import DetectionPlate
import cv2
import requests

app = FastAPI()

@app.get('/')
def status():
    return {"status": "ok"}

@app.post('/detection_plate')
async def search_plate(detection_plate: DetectionPlate):

    image_path = detection_plate.image_path
    confidence = detection_plate.confidence

    # Verificar si la imagen existe
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"La imagen no existe en la ruta: {image_path}"
        )

    # Verificar extensión del archivo
    valid_extensions = ('.png', '.jpg', '.jpeg')
    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de imagen no soportado: {file_extension}. Usa PNG, JPG o JPEG."
        )

    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudo leer la imagen. Verifique que el formato sea válido."
            )

        # Convertir la imagen a bytes en el formato correcto
        success, image_jpg = cv2.imencode(file_extension, image)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="No se pudo convertir la imagen a bytes."
            )

        # Definir el tipo MIME según la extensión
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg"
        }
        mime_type = mime_types[file_extension]

        files = {'upload': ('image' + file_extension, image_jpg.tobytes(), mime_type)}

        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            headers={'Authorization': 'Token 3d307483c1f93feb300793d164dac3a4340e3ec4'},
            files=files
        )

        return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el procesamiento de la imagen: {str(e)}"
        )
