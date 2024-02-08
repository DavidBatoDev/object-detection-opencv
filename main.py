import cv2

#img = cv2.imread("lena.png")  # Load an image
cap = cv2.VideoCapture(0)  # Load a video

cap.set(3, 640)  # Set the width
cap.set(4, 480)  # Set the height
cap.set(10, 100)  # Set the brightness

while True:  # Loop through the video
    success, img = cap.read()  # Read the video
    cv2.imshow("Video", img)  # Display the video
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait for a key press
        break  # Break the loop

# Load the pre-trained model

class_name = []  # Create an empty list to store the class names
class_file = "coco.names"  # Path to the file that contains the class names

with open(class_file, "rt") as f:
    class_name = f.read().rstrip("\n").split("\n")  # Read the class names

config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" # Path to the configuration file

weights_path = "frozen_inference_graph.pb"  # Path to the pre-trained model

net = cv2.dnn_DetectionModel(weights_path, config_path)  # Load the model
net.setInputSize(320, 320)  # Set the input size
net.setInputScale(1.0 / 127.5)  # Set the input scale
net.setInputMean((127.5, 127.5, 127.5))  # Set the input mean
net.setInputSwapRB(True)  # Set the input swapRB

while True:  # Loop through the video
    success, img = cap.read()  # Read the video
    class_ids, confs, bbox = net.detect(img, confThreshold=0.5)  # Detect objects in the image

    if len(class_ids) != 0:  # If there are objects in the image
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):  # Loop through the detected objects
            cv2.rectangle(img, box, (0, 255, 0), 2)  # Draw a rectangle around the object
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Put the confidence on the
            

    cv2.imshow("Output", img)  # Display the image
    cv2.waitKey(0)  # Wait for a key press
