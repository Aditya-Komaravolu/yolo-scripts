import ultralytics
from ultralytics import YOLO
import torch 
# torch.backends.cuda.matmul.allow_tf32 = True

# # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cudnn.allow_tf32 = True

model = YOLO('/home/aditya/yolov10/yolov10x.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/home/aditya/yolov8/snaglist_yolov9_training/yolov9_ultralytics_training_gpt_config_apr20/weights/epoch15.pt')  # load a pretrained model (recommended for training)

# # model.model = torch.compile(model.model, mode="max-autotune")

# # Train the model
results = model.train(
    data='/home/aditya/snaglist_dataset_sep4_high_quality_robo_train_insta_val_multicls/data.yaml', 
    epochs=1000, 
    imgsz=1024, 
    batch=24, 
    device=[0,1],
    patience = 100,
    # exist_ok=True,  #change this to True while resuming
    # resume = True,
    workers = 16,
    optimizer = 'AdamW', #'SGD'
    weight_decay = 2e-3,
    # momentum = None, #0.9
    verbose = True,
    seed = 43,
    cos_lr = True,
    close_mosaic= 0,
    lr0= 0.00001,
    lrf = 0.000001,
    dropout = 0.4,
    val=True,
    plots = True,
    nbs = 64,
    warmup_epochs=3,
    project= "yolov10_training_sep4_high_quality_robo_train_insta_val_multicls",
    name = "yolov10_ultralytics_training_sep4_high_quality_robo_train_insta_val_multicls",
    save_period = 1,
    augment=False,
    single_cls = False ## ENABLE IF ONLY SINGLE CLASS IS THERE IN DATASET OR IF YOU WANT TO CONSIDER ALL CLASSSES AS A SINGLE CLASS.
)


#gpt config
# results = model.train(
#     data='/home/aditya/snaglist_dataset_sep4_high_quality_cement_slurry_yolo/data.yaml', 
#     epochs=1000,  # Set a high epoch as it will be stopped by early stopping
#     imgsz=1024,  # Assuming your dataset has high-res images
#     batch=24,  # Adjust according to your GPU
#     device=0,  # Adjust if you have more than one GPU
#     # patience=10,  # Depends on how quickly your model converges
#     workers=8,  # Adjust as per your CPU
#     optimizer='SGD',  # Or 'Adam' if you prefer
#     weight_decay=1e-2,
#     # resume=True,
#     # exist_ok=True,
#     momentum=0.92,  # Not used with Adam/AdamW
#     close_mosaic=0,
#     verbose=True,
#     seed=42,
#     cos_lr=False,  # Cosine learning rate scheduler
#     lr0=1e-3,  # Starting learning rate
#     lrf=1e-5,  # Final learning rate
#     dropout=0.2,  # Helps prevent overfitting
#     # augment=True,  # Ensure augmentations are applied
#     val=True,
#     plots=True,  # To visualize performance metrics
#     nbs=64,  # Depends on GPU memory
#     warmup_epochs=3,  # Warmup for learning rate
#     project= "yolov10_training_sep4_high_quality_cement_slurry",
#     name = "yolov10_ultralytics_training_sep4_high_quality_cement_slurry",
#     save_period=1  # How often to save the model
# )
