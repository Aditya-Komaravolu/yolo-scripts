
from ultralytics import YOLO

import os
import glob

# Load a pretrained YOLOv8n model
model = YOLO("/home/aditya/yolov10/yolov10_training_sep4_high_quality_robo_train_insta_val_multicls/yolov10_ultralytics_training_sep4_high_quality_robo_train_insta_val_multicls2/weights/best.pt")

# Define path to directory containing images and videos for inference
source = glob.glob("/home/aditya/floor2_25july_frames/*.jpg")

print(source)

output_dir = '/home/aditya/floor2_25july_frames_pred_yolov10_sep4_robo_train_insta_val_multicls'

os.makedirs(output_dir, exist_ok=True)

# Run inference on the source
results = model(
    source,
    conf = 0.3,
    stream=True,
    )  # generator of Results objectsf

print(results)

for i,result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename=f"{output_dir}/image_{i}.jpg")  # save to dis

    del result