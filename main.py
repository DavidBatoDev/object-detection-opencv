import cv2

img = cv2.imread("lena.png")  # Load an image

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

cv2.imshow("Output", img)  # Display the image

cv2.waitKey(0)  # Wait for a key press
