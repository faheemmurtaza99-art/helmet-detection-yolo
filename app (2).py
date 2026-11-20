
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")

st.title("🪖 YOLO Helmet Detection System")
st.write("Upload an image to detect helmets.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(image, conf=conf)

    result_image = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image)

    with col2:
        st.subheader("Detected Image")
        st.image(result_image)

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "Object": model.names[class_id],
            "Confidence": round(confidence, 3)
        })

    st.subheader("Detection Results")

    if detections:
        st.dataframe(pd.DataFrame(detections))
    else:
        st.warning("No objects detected.")
