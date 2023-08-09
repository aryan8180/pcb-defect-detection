import cv2
import numpy as np

def highlight_differences(image1, image2, threshold=30, kernel_size=7):
    # Convert the images to grayscale
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the two grayscale images
    diff = cv2.absdiff(gray_img1, gray_img2)

    # Threshold the difference image to find regions of significant change
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological opening to remove small noise and smoothen the regions
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of the differences
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around the detected differences
    highlighted_img = image2.copy()
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(highlighted_img, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 0, 255), 2)

    return highlighted_img

# Load the two images (replace with your image file paths)
img1 = cv2.imread("ref.png")
img2 = cv2.imread("test.png")

# Resize the images to have the same dimensions (if needed)
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# Highlight the differences between the two images
highlighted_image = highlight_differences(img1, img2)

# Display the highlighted image
cv2.imshow("Highlighted Differences", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
