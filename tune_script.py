from ultralytics import YOLO

# load the pretrained model
model = YOLO("yolo11x.pt")

# perform hyperparameter tuning
tune_results = model.tune(  
                            data='data.yaml',
                            augment=True,
                            hsv_h=0.015,
                            hsv_s=0.7,
                            hsv_v=0.4,
                            degrees=5.0,
                            shear=0.1,
                            translate=0.1,
                            scale=0.5,
                            flipud=0.5,
                            fliplr=0.5,
                            imgsz=640,
                            device=0,
                            batch=5
                        )

print(tune_results)