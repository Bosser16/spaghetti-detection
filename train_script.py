from ultralytics import YOLO

if __name__ == "__main__":
    print("Starting training script...")

    model = YOLO("yolov11s.pt")
    
    # train the model
    train_results = model.train(data='data.yaml', epochs=10, imgsz=640, batch=14, device=0)

    # evaluate performace on the validation set
    metrics = model.val()



