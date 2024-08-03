import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Function to remove prefix from state_dict keys
def remove_prefix(state_dict, prefix):
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

# Load the state dictionary
state_dict = torch.load('vgg16_model_resize(v2).pth', map_location=torch.device('cpu'))
state_dict = remove_prefix(state_dict, 'base_model.')

# Load the model
model = models.vgg16()

# Modify the final classification layer to match the number of classes
num_classes = 3  # Update this to the number of classes your model was trained on
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

# Load the state dictionary
model.load_state_dict(state_dict)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

class_names = ['diabetic_retinopathy', 'normal', 'cataract']

# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes)
#     outputs = model(tensor)
#     probabilities = torch.nn.functional.softmax(outputs, dim=1)
#     confidence, y_hat = probabilities.max(1)
#     return class_names[y_hat.item()], confidence.item()

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, y_hat = probabilities.max(1)
    return class_names[y_hat.item()], confidence.item(), probabilities[0].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'no file'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no file'})
        try:
            img_bytes = file.read()
            prediction, confidence, probabilities = get_prediction(img_bytes)
            
            # Set a threshold for prediction confidence
            confidence_threshold = 0.5

            if confidence <= confidence_threshold:
                result = {
                    'Image': file.filename,
                    'Predicted Class': 'Unknown',
                    'Confidence': confidence,
                    'Class Probabilities': dict(zip(class_names, probabilities))
                }
            else:
                result = {
                    'Image': file.filename,
                    'Predicted Class': prediction,
                    'Confidence': confidence,
                    'Class Probabilities': dict(zip(class_names, probabilities))
                }
            
            # Print the result to console
            print(f"Prediction for {file.filename}: {result['Predicted Class']} with Confidence: {result['Confidence']}")
            
            # Return the result as JSON response
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})



@app.route('/')
def index():
    return 'Welcome to Eye Diseases API!'

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
