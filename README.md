# Object Detection

This is a web app for object detection using different version of [YOLO](https://github.com/ultralytics/yolov5).

![](static/object_detection.gif)

## How it works

The app uses [streamlit](https://github.com/streamlit/streamlit) to create a web interface for [object_detection.py](object_detection.py).

When a user uploads an image, the Python script will detect objects in the image using YOLO and save the results as a new image.

The results are then displayed in the browser.