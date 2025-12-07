import requests
from ultralytics import YOLO
import os
import cv2
import time
import sys
import ctypes
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "http://localhost:5000/api"

# check for -test or -log flags
TEST_MODE = False
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    print("Running in TEST MODE")
    TEST_MODE = True
    
LOG_MODE = False
if len(sys.argv) > 1 and sys.argv[1] == "-log":
    print("Running in LOG MODE")
    LOG_MODE = True
    
# prevent Windows from sleeping while script is running
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

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
# printing = "Printing from SD" or "Printing"
# returns JSON status
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
# if save_image is True, save the image to logs/print_name/
# print_name is used to create a folder for saving images
# return the captured frame
def get_capture(save_image=False, print_name="default"):
    if save_image:
        os.makedirs(f"logs/{print_name}", exist_ok=True)
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    capture.release()
    if save_image:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"logs/{print_name}/captured_image_{timestamp}.jpg", frame)
        print(f"Image saved as captured_image_{timestamp}.jpg")
    return frame
    
# load the trained model
model = YOLO("best.pt")

# if in test mode, just capture one frame and run detection
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
        # if in log mode, save images
        print_name = os.path.splitext(status['job']['file']['name'])[0]
        frame = get_capture(save_image=LOG_MODE, print_name=print_name)
        results = model(frame)
        
        # process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                # in log mode, just print confidences, prevent shutdown
                if LOG_MODE:
                    print(f"Detection confidence: {conf}")
                else:
                    # if any detections with confidence > 0.5, send shutdown command
                    if conf > 0.5:
                        print(f"High confidence detection found: {conf}, sending shutdown command.")
                        shutdown_server()
                        exit(0)
                
                # save annotated image       
                annotated_frame = results[0].plot()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"logs/{print_name}/annotated_image_{timestamp}.jpg", annotated_frame)
                    
        time.sleep(5 * 60)  # wait for 5 minutes before next check