import cv2
import numpy as np
import mss
import os
import urllib.request

# Define the paths to the files
weights_path = "yolov2.weights"
config_path = "yolov2.cfg"
names_path = "coco.names"
reference_image_path = "reference_image.jpg"

# Function to download a file
def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded {url} to {path}")

# Download necessary files if they do not exist
if not os.path.exists(weights_path):
    download_file('https://pjreddie.com/media/files/yolov2.weights', weights_path)

if not os.path.exists(config_path):
    download_file('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg', config_path)

if not os.path.exists(names_path):
    download_file('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', names_path)

if not os.path.exists(reference_image_path):
    print(f"Error: {reference_image_path} not found")
    exit()

# Load YOLOv2
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the reference image
reference_image = cv2.imread(reference_image_path)
ref_height, ref_width, _ = reference_image.shape

# Function to detect objects in the image
def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes, class_ids

# Capture the screen and detect the reference image
def screen_detection(reference_img):
    with mss.mss() as sct:
        while True:
            screen_img = np.array(sct.grab(sct.monitors[1]))
            screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
            boxes, indexes, class_ids = detect_objects(screen_img)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(screen_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(screen_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Screen Detection", screen_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

# Analyze the reference image first
boxes, indexes, class_ids = detect_objects(reference_image)
if len(indexes) > 0:
    print("Reference image detected. Starting screen detection...")
    screen_detection(reference_image)
else:
    print("Reference image not detected.")
