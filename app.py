from flask import Flask, request, render_template
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
model = keras.models.load_model('model.h5')

# Function to preprocess and make predictions
def predict_pneumonia(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(180, 180), color_mode="grayscale")
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)

    # Interpret the prediction
    if prediction > 0.5:
        return "Pneumonia Detected"
    else:
        return "No Pneumonia Detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    # Check if a file was uploaded
    if 'xray_image' not in request.files:
        return "No file part"

    file = request.files['xray_image']

    # Check if the file has a filename
    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded image to the 'uploads' directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Make a prediction
        result = predict_pneumonia(file_path)

        # Pass the image source URL and prediction result to the display.html template
        image_src = 'uploads/' + file.filename
        return render_template('display.html', image_src=image_src, result=result)


if __name__ == '__main__':
    app.run(debug=True)
