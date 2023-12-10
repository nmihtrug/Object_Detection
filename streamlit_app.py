import streamlit as st
from object_detection import object_detection
import os
from datetime import datetime


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

        for model in model_choosens:
            saved_image_path = os.path.join(
                "data",
                file_upload.name.split(".")[0]
                + datetime.now().strftime("%d%m%Y_%H%M%S")
                + f"_{model}."
                + file_upload.name.split(".")[1],
            )

            with open(saved_image_path, "wb") as f:
                f.write(file_upload.getbuffer())

            st.write("Detecting...")
            results = object_detection(saved_image_path, model)

            for subfolder in os.listdir("pred"):
                for file in os.listdir(os.path.join("pred", subfolder)):
                    if file == saved_image_path.split("/")[-1]:
                        st.image(
                            os.path.join("pred", subfolder, file),
                            caption=f"Prediction using {model} model",
                            use_column_width=True,
                        )
                        break


def main():
    st.set_page_config(layout="wide")
    st.title("Object Detection")

    config, prediction = st.columns([2, 3], gap='large')

    with config:
        file_upload, model_choosens = config_container()

    with prediction:
        prediction_container(file_upload, model_choosens)


if __name__ == "__main__":
    main()
