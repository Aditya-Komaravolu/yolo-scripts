import ultralytics
from ultralytics import YOLO


model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/aditya/snaglist_dataset_yolo_mar9/data.yaml', epochs=1000, imgsz=1024, batch=32, device=[0,1])