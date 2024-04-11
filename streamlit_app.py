import streamlit as st
import os
from datetime import datetime
from ultralytics import YOLO
import tempfile
from glob import glob


def config_container():
    model_choosens = st.multiselect(
        "Choose a model", ["yolov8n", "yolov8m", "yolov8l", "yolov8x"]
    )
    file_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if file_upload is not None:
        image = file_upload.read()
        st.image(image, caption="Uploaded Image")
        st.write("")

    return model_choosens, file_upload


def prediction_container(model_choosens, file_upload):
    if file_upload is not None:
        st.write("Detecting...")
        for model_choosen in model_choosens:
            model = YOLO(f"model/{model_choosen}.pt")

            temp_dir = tempfile.TemporaryDirectory()

            uploaded_file_name = file_upload.name
            uploaded_file_path = os.path.join(temp_dir.name, uploaded_file_name)

            with open(uploaded_file_path, "wb") as f0:
                f0.write(file_upload.getbuffer())

            results = model.predict(
                source=uploaded_file_path,
                project=temp_dir.name,
                name="result",
                save=True,
                exist_ok=True,
            )  # save predictions as label
            for f in glob(os.path.join(temp_dir.name, "result", "*")):
                st.image(
                    f,
                    caption=f"Prediction using {model_choosen} model",
                    use_column_width=True,
                )


def main():
    st.set_page_config(layout="wide")
    st.title("Object Detection")

    config, prediction = st.columns([4, 5], gap="large")

    with config:
        file_upload, model_choosens = config_container()

    with prediction:
        prediction_container(file_upload, model_choosens)


if __name__ == "__main__":
    main()
