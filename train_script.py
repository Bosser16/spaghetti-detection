from ultralytics import YOLO
import sys

if __name__ == "__main__":
    print("Starting training script...")
    
    model = YOLO("yolo11x.pt")
    
    # train the model
    train_results = model.train(
        data='data.yaml',
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,
        cos_lr=True,    
        lr0=0.0005,     
        hsv_h=0.1,
        hsv_s=0.7,
        hsv_v=0.7,
        mosaic=1.0,
        mixup=0.2
    )

    # evaluate performance on the validation set
    metrics = model.val()



