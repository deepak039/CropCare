from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
import google.generativeai as genai
from IPython.display import display, Markdown
from PIL import Image
import requests
import textwrap

app = FastAPI()

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True)).data

genai.configure(api_key='AIzaSyBo2OKZF1011lVP9wuOg2YBTaECDSWh6eE')
model = genai.GenerativeModel('gemini-1.5-flash')

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://192.168.35.8:8604/v1/models/plant:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    prompt=f"The potato plant has been diagnosed with {predicted_class}. Please provide suggestions for treatment or care."
    
    response = model.generate_content(prompt)
    suggestion = to_markdown(response.text)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "suggestion": suggestion
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)
