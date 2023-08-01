import cv2

def show_highlighted_differences(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Compute difference
    difference = cv2.subtract(image1, image2)

    # Color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    # Add the red mask to the images to make the differences obvious
    image2[mask != 255] = [0, 0, 255]

    # Display images in separate windows
    cv2.imshow('Orignal Image', image1)
    cv2.imshow('Image with Differences', image2)
    cv2.imshow('Difference Image', difference)

    # Wait for a key event and close the windows when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_highlighted_differences("image.jpg", "temp.jpg")
