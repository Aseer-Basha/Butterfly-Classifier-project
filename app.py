import os
import io
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

# Keras/TensorFlow imports for model loading and image processing
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Configuration ---
UPLOAD_FOLDER = 'static/images/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Assuming the file is in the 'models' folder, as per your structure
MODEL_PATH = 'models/vgg16_model.h5' 

# *Crucial:* Define the Class Names in the EXACT order of your training subfolders
# Your families are: Nymphalidae, Lycaenidae, Hesperiidae
CLASS_NAMES = ['Nymphalidae', 'Lycaenidae'] 


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load the model globally ---
model = None 
try:
    # Set compile=False for models loaded after transfer learning
    model = load_model(MODEL_PATH, compile=False) 
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 


# Function to check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Model Preprocessing Function ---
def preprocess_image(image_file):
    """Loads, resizes, and converts an image into a model-ready array."""
    IMG_SIZE = (224, 224) 
    
    # Load and resize the image
    img = image.load_img(image_file, target_size=IMG_SIZE)
    
    # Convert image to numpy array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize pixel values to 0-1 range
    
    return img_array

# --- Define Routes ---

@app.route('/')
def index():
    """Renders the main landing page (index.html)."""
    return render_template('index.html')

@app.route('/input')
def input_page():
    """Renders the prediction page (input.html)."""
    return render_template('input.html')

@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Handles the newsletter subscription request."""
    email = request.form.get('email')
    
    if email:
        # 1. LOGIC TO SAVE THE EMAIL
        # In a real app, you would save 'email' to a database or file here.
        
        # For now, let's just confirm receipt.
        print(f"New subscription received: {email}")
        
        # 2. REDIRECT OR RETURN SUCCESS
        # You should redirect the user back to the home page or a success page.
        # Example using a redirect:
        return redirect(url_for('index'))
        
    # If the email field was somehow empty
    return jsonify({'error': 'Email not provided'}), 400
@app.route('/predict', methods=['POST'])
def predict():
    """Route to handle the image upload, prediction, and result rendering."""
    # 1. Check Model Status
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    # 2. Check file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # 3. Process the file
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file temporarily
            file.save(filepath)

            # --- PREDICTION LOGIC ---
            processed_image = preprocess_image(filepath) 
            predictions = model.predict(processed_image)[0] # Get the single prediction array
            
            # 1. Prepare list for ALL results
            results_list = []
            for i, family_name in enumerate(CLASS_NAMES):
                confidence = predictions[i] * 100
                results_list.append({
                    'name': family_name,
                    'confidence': f"{confidence:.2f}%"
                })

            # 2. Find the overall BEST prediction
            best_index = np.argmax(predictions)
            best_label = CLASS_NAMES[best_index]
            best_confidence = predictions[best_index] * 100
            
            # Optional: Clean up the saved file after prediction
            os.remove(filepath)
            
            # --- Render Output ---
            return render_template(
                'output.html',
                prediction_result=best_label,      # Main result (highest confidence)
                confidence=f"{best_confidence:.2f}%", # Confidence of the main result
                all_results=results_list,          # NEW: List containing all 3 families/confidences
                image_url=url_for('static', filename=f'images/uploads/{filename}')
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            # Ensure the temporary file is removed even on error if possible
            if os.path.exists(filepath):
                 os.remove(filepath)
            return jsonify({'error': f'Prediction failed. Details: {e}'}), 500
    
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

# --- Run the App ---
if __name__ == '__main__':
    # Runs on http://127.0.0.1:5000/
    app.run(debug=True, host='0.0.0.0', port=5000)