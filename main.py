import numpy as np
import cv2
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
import os
import h5py


app = FastAPI()

LOADED_MODEL = None
CATEGORIES = None
CATEGORIES_RECH = None
MIN_CONFIDENCE = None

LOADED_MODEL = keras.models.load_model(h5py.File(os.environ.get('MODEL',''), 'r'))
    
LOADED_MODEL.summary()

CATEGORIES = os.environ.get('CATEGORIES','').split(',')

MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE","60"))


@app.post('/check-blob-image')
async def check_blob_image(image: UploadFile = File(...)):
    res = {}
    try:
        img = np.asarray(bytearray(await image.read()), dtype="uint8")
        img = cv2.imdecode(img,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img,(96,96))
        img = img.astype("float") / 255.0
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        pred = LOADED_MODEL.predict(img)[0]
        pred_ix = np.argmax(pred, axis=0)
        pred_tag = CATEGORIES[pred_ix]
        confidence = round(pred[pred_ix] * 100, 1)
        res["value"] = {'type': pred_tag, 'confidence': confidence}
        res["error"] = None
        res["min_confidence"] = MIN_CONFIDENCE
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

