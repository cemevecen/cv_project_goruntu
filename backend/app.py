#full kod
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = FastAPI()

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "model.h5"
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Preprocess image for MobileNetV2
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    # Formatting result
    results = []
    for imagenet_id, label, prob in decoded:
        results.append({"label": label, "probability": float(prob)})

    return {"prediction": results[0]["label"], "details": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)