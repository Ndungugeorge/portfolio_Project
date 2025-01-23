from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from oauthlib.oauth2 import WebApplicationClient
import requests
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth setup
GOOGLE_CLIENT_ID = "your_google_client_id"
GOOGLE_CLIENT_SECRET = "your_google_client_secret"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

FACEBOOK_CLIENT_ID = "your_facebook_client_id"
FACEBOOK_CLIENT_SECRET = "your_facebook_client_secret"
FACEBOOK_OAUTH_URL = "https://www.facebook.com/v12.0/dialog/oauth"

google_client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Define the folder paths for model and static files
MODEL_DIR = os.path.join(app.root_path, 'models')
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'upload')
STATIC_IMAGE_FOLDER = os.path.join(app.root_path, 'static', 'images')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
model = load_model(MODEL_PATH)
print("Model Loaded Successfully")

# User model for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    name = db.Column(db.String(150), nullable=False)
    password = db.Column(db.String(150))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Prediction function
def predict_disease(image_path):
    test_image = load_img(image_path, target_size=(128, 128))
    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    pred = np.argmax(result, axis=1)[0]
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

# Routes
@app.route('/')
@login_required
def home():
    return render_template('main.html', name=current_user.name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
        else:
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return redirect(url_for('home'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    pred, output_page = predict_disease(file_path)
    return render_template(output_page, pred_output=pred, user_image=file_path)

# Google OAuth
@app.route('/login/google')
def google_login():
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    request_uri = google_client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route('/login/google/callback')
def google_callback():
    code = request.args.get("code")
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]
    token_url, headers, body = google_client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )
    google_client.parse_request_body_response(token_response.text)
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = google_client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)
    user_data = userinfo_response.json()
    email = user_data["email"]
    name = user_data["name"]
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, name=name)
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(threaded=False, port=8080)
