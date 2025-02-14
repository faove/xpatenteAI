from fastapi import FastAPI, HTTPException
import os.path
from models.DetectionPlate import DetectionPlate
from ultralytics import YOLO
import cv2
import pytesseract

app = FastAPI()

model = YOLO('best.pt')

@app.get('/')

def status():
    return true

@app.post('/detection_plate')

async def search_plate(detection_plate: DetectionPlate):

    image_path = detection_plate.image_path
    confidence = detection_plate.confidence
    
    #image_path = '/var/www/html/xdetectionplate/storage/app/public/patentes/01JK6PTXTR73NWEC471FJ0ZPFV.jpg'

    # image_pathg = image_path.replace('\\', '/')   
    
    
    # Verificar si la imagen existe
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"La imagen no existe en la ruta: {image_path}"
        )

    try:
        # Intentar leer la imagen
        image = cv2.imread(image_path)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudo leer la imagen. Verifique que el formato sea válido."
            )

        results = model.predict(source=image_path, conf=confidence)

        
        if len(results[0].boxes) == 0:
            return {"error": "No se detectó ninguna placa"}

        # Cargar la imagen original
        image = cv2.imread(image_path)

        plates = []
        
        for box in results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordenadas de la placa

            plate_crop = image[y1:y2, x1:x2]  # Recortar la placa detectada
            # Convertir imagen a escala de grises
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # Aplicar filtro bilateral para reducir el ruido y mantener los bordes
            # plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

            # Aplicar umbral adaptativo para mejorar el contraste
            # plate_thresh = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #                                     cv2.THRESH_BINARY, 11, 2)

            # Guardar la imagen procesada para verificar visualmente
            #cv2.imwrite("plate_processed.jpg", plate_thresh)
            # cv2.imwrite("plate_crop_2.jpg", plate_crop)
            #return box.show()
    
            # Convertir imagen para OCR
            # plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            # _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Extraer texto con OCR
            #plate_text = pytesseract.image_to_string(plate_thresh, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            # Aplicar un filtro Gaussiano para suavizar
            plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 1)
            #plate_gray = cv2.medianBlur(plate_gray, 9)
            

            # Usar un umbral simple en lugar de umbral adaptativo
            _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Configuración mejorada de OCR
            custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            # custom_config = r'--psm 6 --oem 3'
            plate_text = pytesseract.image_to_string(plate_thresh, config=custom_config)
            # custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            # plate_text = pytesseract.image_to_string(plate_thresh, config=custom_config)      # Modo de segmentación óptimo para una línea de texto

            plates.append({
                "box": [x1, y1, x2, y2],
                "plate_text": plate_text.strip()
            })

        return {"detected_plates": plates}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el procesamiento de la imagen: {str(e)}"
        )