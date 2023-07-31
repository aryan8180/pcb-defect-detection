import cv2
import numpy as np

def create_heatmap(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Compute absolute pixel-wise difference
    difference = cv2.absdiff(image1, image2)

    # Normalize the difference image to [0, 255]
    difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply a color map to create a heatmap
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)

    # Display the heatmap
    cv2.imshow('Difference Heatmap', heatmap)

    # Wait for a key event and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace "left.jpg" and "right.jpg" with the paths to your desired images
    create_heatmap("left.jpg", "right.jpg")
