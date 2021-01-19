import numpy as np
import cv2
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
import os
import h5py

app = FastAPI()

LOADED_MODEL = keras.models.load_model(h5py.File(os.environ.get('MODEL',''), 'r'))
LOADED_MODEL.summary()
CATEGORIES = os.environ.get('CATEGORIES','').split(',')

@app.post('/check-blob-image')
async def check_blob_image(image: UploadFile = File(...)):
    res = {}
    try:
        ''' Primero cargamos la foto desde form data hacia numpy'''
        img = np.asarray(bytearray(await image.read()), dtype="uint8")
        ''' Cargamos la foto en formato correcto en open-cv'''
        img = cv2.imdecode(img,cv2.IMREAD_UNCHANGED)
        ''' Estandarizamos la imagen de acuerdo al tamaño de imagen con el que entrenamos el modelo'''
        img = cv2.resize(img,(96,96))
        ''' Normalización de la foto'''
        img = img.astype("float") / 255.0
        img = np.array(img)
        ''' Formato de entrada hacia red neuronal (96,96,3)  => (1,96, 96,3 )'''
        img = np.expand_dims(img, axis=0)
        ''' Tomamos el modelo y ejecutamos la predicción'''
        pred_arr = LOADED_MODEL.predict(img)
        ''' Keras nos retornará la probabilidad por cada categoría '''
        print('prediction:')
        print(pred_arr)
        pred = pred_arr[0]
        ''' Retornaremos la predicción más alta '''
        pred_ix = np.argmax(pred, axis=0)
        pred_tag = CATEGORIES[pred_ix]
        confidence = round(pred[pred_ix] * 100, 1)
        res["value"] = {'type': pred_tag, 'confidence': confidence}
        res["error"] = None
        return res
    except Exception as exception:
        res["validator"] = None
        res["value"] = None
        res["error"] = "Internal error"
        print(str(exception))
        return res

@app.get("/")
def home():
    return {"status": "SERVICIO CORRIENDO"}        

