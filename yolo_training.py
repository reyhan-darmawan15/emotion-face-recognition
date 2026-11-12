# Script for training the YOLO26 model
# Since the use-case is for face expression image classification, we will use YOLO26l-cls model

import datetime
from ultralytics import YOLO

if __name__ == "__main__":
    # Load the new model
    model = YOLO('yolo26x-cls.yaml')
    print("YOLO26x-cls model loaded.")

    # Start training
    print("Training started at " + str(datetime.datetime.now()))
    result = model.train(data="./datasets", 
                         project=".", 
                         cache=True, 
                         epochs=500, 
                         exist_ok=True, 
                         imgsz=128, 
                         device=0, 
                         val=True, 
                         plots=True, 
                         pretrained=False, 
                         optimizer='MuSGD', 
                         patience=0, 
                         cos_lr=True)
    print("Training finished at " + str(datetime.datetime.now()))

    # Validate the result
    metrics = model.val(exist_ok=True)
    print("Validation finished.")