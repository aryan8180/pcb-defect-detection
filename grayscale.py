import cv2
import numpy as np

def create_difference_image(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Compute absolute pixel-wise difference
    difference = cv2.absdiff(image1, image2)

    # Display the grayscale difference image
    cv2.imshow('Grayscale Difference Image', difference)

    # Wait for a key event and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace "left.jpg" and "right.jpg" with the paths to your desired images
    create_difference_image("left.jpg", "right.jpg")
