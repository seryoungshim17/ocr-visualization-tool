from fastapi import FastAPI, File, UploadFile, Response, Form
import uvicorn
import numpy as np
import cv2
from utils import load_weight
from models.modules.utils.util import predict

# FastAPI 객체 생성
app = FastAPI()

@app.post("/")
async def read_root(file: UploadFile = File(...), model: str = Form(...), weight: str = Form(...)):
    model = load_weight(model, weight)
    
    print(weight)
    content = await file.read()
    encoded_img = np.fromstring(content, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    prediction = predict(img, model)
    
    _, encoded = cv2.imencode('.jpg', prediction)
    return Response(encoded.tobytes(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)