import requests
from ultralytics import YOLO
import os
import cv2
import time
import sys
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "http://localhost:5000/api"

TEST_MODE = False
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    print("Running in TEST MODE")
    TEST_MODE = True

# shutdown the print server
def shutdown_server():
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": API_KEY
    }

    response = requests.post(
        f"{BASE_URL}/job",
        headers=headers,
        json={"command": "cancel"}
    )

    print(response.status_code, response.json())

# Check server status    
# idle = "Operational"
# printing = 'Printing from SD' or "Printing"
def print_server_status():
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": API_KEY
    }
    response = requests.get(
        f"{BASE_URL}/job",
        headers=headers
    )
    print(response.status_code, response.json())
    return response.json()
    
# capture an image from the default camera
def get_capture(save_image=False):
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    capture.release()
    if save_image:
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved as captured_image.jpg")
    return frame
    
    
model = YOLO("best.pt")

if TEST_MODE:
    frame = get_capture(save_image=True)
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"Class: {cls}, Confidence: {conf}, Box: {xyxy}")
            
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    while True:
        # check if server is printing, if not, exit loop
        status = print_server_status()
        if status['state'] not in ["Printing", "Printing from SD"]:
            print("Server is not printing, exiting loop.")
            break
        
        # capture frame and run obj detection
        frame = get_capture(save_image=False)
        results = model(frame)
        
        # if any detections with confidence > 0.5, send shutdown command
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                if conf > 0.5:
                    print(f"High confidence detection found: {conf}, sending shutdown command.")
                    shutdown_server()
                    exit(0)
                    
        time.sleep(5 * 60)  # wait for 5 minutes before next check