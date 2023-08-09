import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def resize_image(image, width, height):
    # Resize the image to the specified width and height
    return cv2.resize(image, (width, height))

def capture_image_from_camera():
    # Access the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Camera', frame)

        # Wait for a key event (Press space bar to capture the image)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Release the camera and close the window
            cap.release()
            cv2.destroyAllWindows()

            # Return the captured image (resize it to a specific resolution)
            return resize_image(frame, 640, 480)  # Adjust width and height as needed

def show_highlighted_differences(image1, image2):
    # Compute difference
    difference = cv2.subtract(image1, image2)

    # Color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    # Add the red mask to the images to make the differences obvious
    image2[mask != 255] = [0, 0, 255]

    # Display images in separate windows
    cv2.imshow('Reference Image', image1)
    cv2.imshow('Image from Camera', image2)
    cv2.imshow('Difference Image', difference)

    # Compare the images using SSIM and MSE
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ssim_score, _ = ssim(gray_image1, gray_image2, full=True)
    mse = np.mean((image1 - image2) ** 2)

    print(f"SSIM Score: {ssim_score:.4f}")
    print(f"MSE Value: {mse:.2f}")

    ssim_threshold = 0.7
    mse_threshold = 1000

    if ssim_score >= ssim_threshold:
        print("The images are similar based on SSIM.")
    else:
        print("The images are different based on SSIM.")

    if mse <= mse_threshold:
        print("The images are similar based on MSE.")
    else:
        print("The images are different based on MSE.")

    # Wait for a key event and close the windows when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Capture an image from the camera and resize it to a specific resolution
    camera_image = capture_image_from_camera()

    # Load the reference image and resize it to the same resolution
    reference_image = cv2.imread("image.jpg")
    reference_image = resize_image(reference_image, 640, 480)  # Adjust width and height as needed

    # Compare the captured camera image with the reference image and show differences
    show_highlighted_differences(reference_image, camera_image)
