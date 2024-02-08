import cv2

img = cv2.imread("lena.png")  # Load an image

class_name = []  # Create an empty list to store the class names
class_file = "coco.names"  # Path to the file that contains the class names

with open(class_file, "rt") as f:
    class_name = f.read().rstrip("\n").split("\n")  # Read the class names

print(class_name)  # Print the class names

cv2.imshow("Output", img)  # Display the image

cv2.waitKey(0)  # Wait for a key press
