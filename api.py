from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Allow all file types
def allowed_file(filename):
    return True

# Load ResNet-18 model with 4 output classes
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Assuming 4 output classes

# Load saved model weights
checkpoint = torch.load('resnet18_model.pth', map_location=torch.device('cpu'))

# Filter the state dictionary to only include matching keys
model_dict = model.state_dict()
checkpoint_filtered = {k: v for k, v in checkpoint.items() if k in model_dict}

# Load the filtered state dictionary into the model
model.load_state_dict(checkpoint_filtered, strict=False)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing function
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Define the prediction endpoint for the model
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Read the file and convert it to bytes
        image_bytes = file.read()

        # Preprocess the image
        image_tensor = preprocess_image(image_bytes)

        # Make a prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply softmax to output to get prediction probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
            prediction = int(predicted[0])

        # Define your own class labels
        class_labels = ['glaucoma', 'diabetic_retinopathy', 'normal', 'cataract']
        predicted_label = class_labels[prediction]

        # Convert probabilities to a list and then to JSON serializable format
        probabilities = probabilities.tolist()

        # Return the prediction and probabilities
        return jsonify({
            'prediction': predicted_label, 
            'probabilities': probabilities
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
@app.route('/')
def index():
    return 'Welcome to Eye Diseases API!'

if __name__ == '__main__':
    app.run(debug=True)
