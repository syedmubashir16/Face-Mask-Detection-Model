import streamlit as st
import torch
from PIL import Image

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
    return model

model = load_model()

st.title("😷 Face Mask Detection App")
st.write("Upload an image to check if a person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    st.write("### Detection Results")
    st.image(results.render()[0])
