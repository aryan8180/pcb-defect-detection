from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def show_highlighted_differences(image1, image2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]

    _, img1_png = cv2.imencode('.png', image1)
    image1_base64 = base64.b64encode(img1_png).decode('utf-8')

    _, img2_png = cv2.imencode('.png', image2)
    image2_base64 = base64.b64encode(img2_png).decode('utf-8')

    _, diff_png = cv2.imencode('.png', difference)
    difference_base64 = base64.b64encode(diff_png).decode('utf-8')

    return image1_base64, image2_base64, difference_base64

@app.route('/', methods=['GET', 'POST'])
def display_images():
    if request.method == 'POST':
        if 'input_image' not in request.files or 'reference_image' not in request.files:
            return render_template('index.html', error="Please upload both input and reference images.")

        input_image = request.files['input_image']
        reference_image = request.files['reference_image']

        if input_image.filename == '' or reference_image.filename == '':
            return render_template('index.html', error="Please select both input and reference images.")

        if input_image and allowed_file(input_image.filename) and reference_image and allowed_file(reference_image.filename):
            input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], input_image.filename)
            reference_img_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_image.filename)

            input_image.save(input_img_path)
            reference_image.save(reference_img_path)

            image1, image2, difference = show_highlighted_differences(input_img_path, reference_img_path)

            return render_template('result.html', image1=image1, image2=image2, difference=difference)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
