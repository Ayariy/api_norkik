from fastapi import FastAPI
from fastapi import File
import uvicorn
import pytesseract as tess
import numpy as np
from io import BytesIO

import cv2

app = FastAPI()

tess.pytesseract.tesseract_cmd = r'/app/.apt/usr/bin/tesseract'
#METODOS DE LA API ADICIONALES LAS RESPUESTAS 


def read_img(img):
 text = tess.image_to_string(img)
 return(text)

@app.get('/index')
def hello(name:str):
    return f'hello {name}'



@app.post('/api/ocr')
async def ocrFuntion(file: bytes = File(...)):
    image_stream = BytesIO(file)

    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    label = read_img(frame)

    return {'text': label}



if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='localhost')