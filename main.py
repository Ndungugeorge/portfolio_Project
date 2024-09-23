from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Define the folder paths for model and static files
MODEL_DIR = os.path.join(app.root_path, 'models')  # Model directory relative to the app root
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'upload')  # Upload directory inside 'static'
STATIC_IMAGE_FOLDER = os.path.join(app.root_path, 'static', 'images')  # Static images folder

# Load the model from the models folder
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
model = load_model(MODEL_PATH)
print("Model Loaded Successfully")

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Prediction function
def predict_disease(image_path):
    test_image = load_img(image_path, target_size=(128, 128))  # Load and resize image
    print("@@ Got Image for prediction")
    
    test_image = img_to_array(test_image) / 255.0  # Normalize the image
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    
    # Perform the prediction
    result = model.predict(test_image)
    print('@@ Raw result =', result)
    
    # Get the predicted class index
    pred = np.argmax(result, axis=1)[0]

    # Mapping prediction index to disease name and HTML file
    disease_map = {
        0: ("Tomato - Bacteria Spot Disease", 'Tomato-Bacteria_Spot.html'),
        1: ("Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'),
        2: ("Tomato - Healthy and Fresh", 'Tomato-Healthy.html'),
        3: ("Tomato - Late Blight Disease", 'Tomato-Late_Blight.html'),
        4: ("Tomato - Leaf Mold Disease", 'Tomato-Leaf_Mold.html'),
        5: ("Tomato - Septoria Leaf Spot Disease", 'Tomato-Septoria_Leaf_Spot.html'),
        6: ("Tomato - Target Spot Disease", 'Tomato-Target_Spot.html'),
        7: ("Tomato - Tomato Yellow Leaf Curl Virus", 'Tomato-Tomato_Yellow_Leaf_Curl_Virus.html'),
        8: ("Tomato - Tomato Mosaic Virus", 'Tomato-Tomato_Mosaic_Virus.html'),
        9: ("Tomato - Two Spotted Spider Mite", 'Tomato-Two_Spotted_Spider_Mite.html')
    }
    
    return disease_map.get(pred, ("Unknown Disease", 'none1.html'))

# Route for the home page
@app.route("/", methods=['GET'])
def home():
    return render_template('main.html')

# Route to handle image upload and prediction
@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save the uploaded file to the upload folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    print(f"@@ Input posted = {file.filename}")
    print("@@ Predicting class...")

    # Get prediction and corresponding HTML file
    pred, output_page = predict_disease(file_path)
    
    # Serve the prediction result
    return render_template(output_page, pred_output=pred, user_image=file_path)

# For running on local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080)
