from ultralytics import YOLO


def object_detection(image_path, model):
    # Load a model
    model = YOLO(f"model/{model}.pt")  # load a pretrained model (recommended for training)

    results = model.predict(source=image_path, project='pred', save = True)  # save predictions as label
    return results

# object_detection('data/bus08_12_2023__16_37_25.jpg', 'yolov8x')