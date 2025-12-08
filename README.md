# spaghetti-detection

## tune_script.py
This file performs hyper training on the dataset. It predicts the best hyper parameters for training.

## train_script.py
This file trains a YOLO model using the hyper parameters found in the tuning. There results and best version of the model are saved to a runs/ folder

## model_sample.ipynb
This jupyter notebook is a demonstration for this submission. This file will loop through images in sample_images/ and display the results from the trained model.

## shutdown.py
This is the main file for this project. It runs and model and is what communicates with my printer server. I was able to set up my printer to be able to talk to my computer on a locally hosted web server. I can check the status of the printer as well as send commands like cancel, pause, and resume. This file is also able to access my camera. I mounted my web cam to the side of the printer so I can periodically capture the printing process. There are three modes this file can run in, test mode and log mode with the -test and -log flag respectively. Normal mode is ran with no flags.
#### NORMAL MODE
In normal mode, run with no flags, the code will capture an image every 5 minutes as long as the printer is currently in a printing state. Once captured, the image is processed by the model and the results are analyzed. If any spaghetti errors were detected above the decided confidence level, a shutdown command is sent to the print server to stop the print.
#### TEST MODE
In test mode, only one image is captured and analyzed. The results are displayed in a pop up on screen. This mode is to test to make sure the model, camera, printer communication interface, and images are all working as intended.
#### LOG MODE
In log mode, the code captures every 5 minutes like normal, but every image captured is saved in a log folder. If any detections are made, these annotated images are also saved. If an error is detected above the confidence level, no shutdown command is sent. This mode is used to gather image data for further training. The shutdown command is not sent, so more error images can be saved and manually analyzed later. 