import cv2
import inference
import supervision as sv
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

annotator = sv.BoxAnnotator()


def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.4]
    print(detections)
    cv2.imshow(
        "Prediction",
        annotator.annotate(scene=image, detections=detections, labels=labels),
    ),
    cv2.waitKey(1)


inference.Stream(
    source="webcam",  # or rtsp stream or camera id
    model="banana-ripening-process/2",  # from Universe
    output_channel_order="BGR",
    use_main_thread=True,  # for opencv display
    on_prediction=on_prediction,
    api_key=API_KEY,
)
