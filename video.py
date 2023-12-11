import cv2
import inference
import os
import supervision as sv
import numpy as np

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ROBOFLOW_API_KEY')

from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("banana-ripening-process")
model = project.version(2).model

job_id, signed_url, _ = model.predict_video(
    "banana.webm",
    fps=5,
    prediction_type="batch-video",
    additional_models=["clip"]
)

results = model.poll_until_video_results(job_id)

def callback(scene: np.ndarray, index: int) -> np.ndarray:
    results = results[index]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        results.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=scene, detections=detections)

    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image