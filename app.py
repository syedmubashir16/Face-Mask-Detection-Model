import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (replace with your trained weights path)
@st.cache_resource
def load_model(model_path="models/best.pt"):
    return YOLO(model_path)

def main():
    st.title("😷 Face Mask Detection with YOLOv8")
    st.sidebar.title("Settings")
    st.sidebar.markdown("---")

    # Sidebar app mode selection
    app_mode = st.sidebar.selectbox(
        "Choose the App Mode",
        ["About App", "Run on Image", "Run on Video"]
    )

    # About section
    if app_mode == "About App":
        st.markdown("""
        This app uses a **YOLOv8** model trained to detect whether a person is 
        wearing a face mask or not.  
        Built with **Streamlit** for deployment and easy interaction.
        """)
        st.markdown("""
        ### Classes:
        - Mask 😷
        - No Mask 🙅‍♂️
        """)

    # Run on Image
    elif app_mode == "Run on Image":
        confidence = st.sidebar.slider("Confidence", min_value=0.15, max_value=1.0, value=0.5)
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        demo_image = "sample_dataset/demo.jpg"

        if img_file_buffer is not None:
            img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(demo_image)
            image = np.array(Image.open(demo_image))

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        # Load model
        model = load_model()
        results = model.predict(img, conf=confidence)

        # Show results
        st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

    # Run on Video
    elif app_mode == "Run on Video":
        conf = st.sidebar.slider("Confidence", min_value=0.25, max_value=1.0, value=0.5)
        use_webcam = st.sidebar.checkbox("Use Webcam")
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "asf"])

        demo_video = "sample_dataset/demo.mp4"
        tffile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0  # webcam
            else:
                tffile.name = demo_video
                demo_vid = open(tffile.name, "rb")
                demo_bytes = demo_vid.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, "rb")
            demo_bytes = demo_vid.read()
            st.sidebar.text("Input Video")
            st.sidebar.video(demo_bytes)

        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html=True)

        # Load model
        model = load_model()

        cap = cv2.VideoCapture(tffile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise Exception(e)
