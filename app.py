# app.py
# ------------------------------------------------------------
# This is the main file that runs your Hugging Face Space.
# It loads your Teachable Machine model and shows a simple web
# page where anyone can upload a photo and see the prediction.
# You do not need to change anything in this file.
# ------------------------------------------------------------

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import gradio as gr

# ---- 1. Load the model and class names once, when the app starts ----

# Loads the Keras model file you exported from Teachable Machine.
# "compile=False" is used because we only need the model for predictions,
# not for further training.
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Reads the class names from labels.txt.
# Each line in that file looks like: "0 mobile"
# We only keep the text part after the number.
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f.readlines() if line.strip()]

# Teachable Machine image models always expect a 224x224 pixel image.
IMAGE_SIZE = (224, 224)


def predict(image):
    """
    Runs every time someone uploads a photo.
    Returns a dictionary like {"mobile": 0.92, "watch": 0.05, "face": 0.03}
    which Gradio automatically displays as a labeled bar chart.
    """
    if image is None:
        return None

    # Make sure the image has 3 color channels (Red, Green, Blue)
    image = image.convert("RGB")

    # Resize and crop the image to exactly 224x224, the size the model expects
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array of numbers
    image_array = np.asarray(image)

    # Teachable Machine models expect pixel values between -1 and 1,
    # not the usual 0-255, so we convert them here.
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # The model expects a "batch" of images (even if it's just 1),
    # so we add an extra dimension at the front.
    data = np.expand_dims(normalized_image_array, axis=0)

    # Run the actual prediction
    predictions = model.predict(data)[0]

    # Turn the raw numbers into a {class_name: confidence} dictionary
    results = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return results


# ---- 2. Build the web page (Gradio interface) ----

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=len(class_names), label="Prediction"),
    title="My Teachable Machine Image Classifier",
    description=(
        "Upload a photo and the model will predict which of these classes it "
        "belongs to: " + ", ".join(class_names) + ". "
        "This is a personal/classroom demo project, not a certified tool."
    ),
)

# ---- 3. Start the app ----
if __name__ == "__main__":
    demo.launch()
