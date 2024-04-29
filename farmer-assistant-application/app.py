from flask import Flask, render_template, request
import pickle
import subprocess
import numpy as np
import os
import base64
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from ultralytics import YOLO
import io
from werkzeug.utils import secure_filename
from PIL import Image
app = Flask(__name__)

NB_pkl_filename = "NBClassifier.pkl"  
NB_model = None

def load_custom_model():
    global NB_model
    if NB_model is None:
        with open(NB_pkl_filename, 'rb') as f:
            NB_model = pickle.load(f)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "grp17"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class_names = ["Pepper_bell_Bacterial_spot",
 "Pepper_bell_healthy",
 "Potato_EarlyBlight",
 "Potato_Healthy",
 "Potato_LateBlight"]

model = load_model("model.h5")  


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_streamlit')
def run_streamlit():

    streamlit_app_folder = './chatbot'
    os.chdir(streamlit_app_folder)


    subprocess.run(['streamlit', 'run', 'chatbot_app.py'])

    return 'Streamlit app executed'

@app.route('/disease_recognition')
def disease_recognition():
    return render_template('disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        crop_disease_remedies = {
    "Pepper_bell_Bacterial_spot": "Select resistant varieties \n Purchase disease-free seeds and transplants. \n Treat seeds by soaking them for 2 minutes in a 10% chlorine bleach solution (1 part bleach; 9 parts water). Thoroughly rinse seeds and dry them before planting. \n Mulch plants deeply with a thick organic material like newspaper covered with straw or grass clippings. \n Avoid overhead watering. \n Remove and discard badly infected plant parts and all debris at the end of the season. \n Spray every 10-14 days with fixed copper (organic fungicide) to slow down the spread of infection.\n Rotate peppers to a different location if infections are severe and cover the soil with black plastic mulch or black landscape fabric prior to planting.",
    "Pepper_bell_healthy": "Water your healthy Pepper Bell plant consistently to maintain its vigor.",
    "Potato_EarlyBlight": "To deal with Potato Early Blight on your farm, start by cutting off any potato leaves that look sick and  burn it rather than compost it. This helps stop the disease from spreading.  You can also use a special spray (fungicide) to protect your potatoes from getting more sick. When you water your plants, make sure the water doesn't touch the leaves; it should go to the ground. Give your plants enough space so they can breathe, and put some natural stuff (mulch) around them to keep the soil from splashing onto the leaves.",
    "Potato_Healthy": "Keep up the good work! Your potatoes look healthy.",
    "Potato_LateBlight" : "Remove affected leaves, use the right fungicide, and avoid wetting leaves when watering. Provide space for air circulation, use mulch, and don't plant potatoes in the same spot next year. Choose resistant potato varieties, keep a close eye on your plants, and dispose of infected material properly. These steps can protect your potato crop from late blight."
}
        if 'file' not in request.files:
            return render_template('disease.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file
        if file.filename == '':
            return render_template('disease.html', error='No selected file')

        # If the file is a valid image
        if file and allowed_file(file.filename):
            # Process the image
            img = Image.open(io.BytesIO(file.read()))
            img = img.convert('RGB')
            img = img.resize((256, 256))
            img_array = image.img_to_array(img)
            img_array = np.array([img_array])

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)
            remedy = crop_disease_remedies.get(predicted_class, "Remedy information not available")

            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(file_path, format='JPEG') 
            return render_template('disease.html', prediction=predicted_class, confidence=confidence, filename=filename, remedy=remedy)

        else:
            return render_template('disease.html', error='Invalid file type')

    except Exception as e:
        return render_template('disease.html', error=f'Error processing image: {str(e)}')

def allowed_file(filename):
    # Check if the file has a valid extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

    
@app.route('/gov_form')
def gov_form():
    return render_template('gov.html')

@app.route('/predict_gov', methods=['POST'])
def predict_gov():
    # Load the Naive Bayes model if not already loaded
    load_custom_model()

    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        citizen = request.form['citizen']
        landowner = request.form['landowner']
        bankaccount = request.form['bankaccount']
        landsize = float(request.form['landsize'])
        farmertype = request.form['farmertype']
        income = int(request.form['income'])
        coborrower = request.form['coborrower']
        kcc = request.form['kcc']

        # Preprocess categorical data
        citizen_encoded = 1 if citizen == 'yes' else 0
        landowner_encoded = 1 if landowner == 'yes' else 0
        bankaccount_encoded = 1 if bankaccount == 'yes' else 0
        coborrower_encoded = 1 if coborrower == 'yes' else 0
        kcc_encoded = 1 if kcc == 'yes' else 0

        # Map farmer type selection to the expected encoding (1, 2, 3)
        farmertype_mapping = {"small & marginal": 3, "cultivator": 2, "sharecropper": 1}
        farmertype_encoded = farmertype_mapping[farmertype]

        # Prepare the feature vector for prediction
        features = [age, citizen_encoded, landowner_encoded, bankaccount_encoded, landsize, farmertype_encoded, income, coborrower_encoded, kcc_encoded]

        # Make prediction using the loaded Naive Bayes model
        prediction = NB_model.predict([features])

        # Assuming the model predicts the scheme name
        predicted_scheme = prediction[0]

        return render_template('result.html', prediction=predicted_scheme)  # Replace with your prediction display template

weed_model = YOLO('best.pt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def predict_on_image(image_stream):
    image = cv2.imdecode( np.asarray(bytearray(image_stream.read()), dtype=np.uint8) , cv2.IMREAD_COLOR)

    results = weed_model.predict(image, classes=1, conf=0.25)
    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)

    return im_bgr

@app.route('/weed')
def weed():
    return render_template('weed.html')

@app.route('/weed_det', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('weed.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('weed.html', error='No selected file')

        if file and allowed_file(file.filename):

            predicted_image = predict_on_image(file.stream)

            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')

            return render_template('weed_result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

    return render_template('weed.html')

if __name__ == '__main__':
    app.run(debug=True)
