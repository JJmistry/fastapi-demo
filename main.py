from fastapi import FastAPI, File, UploadFile
import io

from image_classification import load_densenet, process_image_for_tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# load model
model = load_densenet()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is running"}


# Post request which calls model
@app.post("/predict", status_code=200)
def predict(image_file: UploadFile = File(...)):

    # retrieve byte stream and convert to image
    image_bytes = image_file.file.read()
    image_stream = io.BytesIO(image_bytes)

    # Convert image to required format to make predictions
    img_std = process_image_for_tf(image_stream)

    # call prediction function
    preds = model.predict(img_std)

    # decode predictions from model and format as json-friendly dict
    decoded_preds = decode_predictions(preds, top=1)[0][0][1]
    results = {"response": decoded_preds}

    return results
