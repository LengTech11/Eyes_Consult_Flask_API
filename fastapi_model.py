import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import uvicorn # type: ignore

warnings.filterwarnings("ignore")

app = FastAPI()

# Define EyeOrNotModel class
class EyeOrNotModel(torch.nn.Module):
    def __init__(self):
        super(EyeOrNotModel, self).__init__()
        self.base_model = models.vgg16(pretrained=False)
        self.base_model.classifier[6] = torch.nn.Linear(self.base_model.classifier[6].in_features, 2)  # 2 classes: 'no_eye' and 'eye'

    def forward(self, x):
        return self.base_model(x)

# Define VGG model for eye diseases
def remove_prefix(state_dict, prefix):
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

# Initialize and load the EyeOrNotModel
eye_or_not_model = EyeOrNotModel()
eye_or_not_model.load_state_dict(torch.load('eye_or_not_eye1.pth', map_location=torch.device('cpu')))
eye_or_not_model.eval()

# Initialize and load the VGG16 model for eye diseases
vgg16_model = models.vgg16(pretrained=False)
num_classes = 4  # Update this to the number of classes for VGG model
vgg16_model.classifier[6] = torch.nn.Linear(vgg16_model.classifier[6].in_features, num_classes)
state_dict = torch.load('vgg16_model_resize_v1.2(4classes).pth', map_location=torch.device('cpu'))
state_dict = remove_prefix(state_dict, 'base_model.')
vgg16_model.load_state_dict(state_dict)
vgg16_model.eval()

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

# Define class names for the VGG model and EyeOrNotModel
vgg_class_names = ['diabetic_retinopathy', 'normal', 'glaucoma', 'cataract']
eye_class_names = ['not_eye', 'eye']

def get_vgg_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = vgg16_model(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, y_hat = probabilities.max(1)
    return vgg_class_names[y_hat.item()], confidence.item(), probabilities[0].tolist()

def get_eye_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = eye_or_not_model(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, y_hat = probabilities.max(1)
    y_hat_index = y_hat.item()

    if y_hat_index >= len(eye_class_names):
        raise ValueError("Index for eye class names is out of range")

    return eye_class_names[y_hat_index], confidence.item(), probabilities[0].tolist()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        img_bytes = await file.read()
        eye_prediction, eye_confidence, eye_probabilities = get_eye_prediction(img_bytes)

        if eye_prediction == 'not_eye':
            # If predicted 'not_eye', return a direct response
            result = {
                'Image': file.filename,
                'Predicted Class': 'Not Eye',
                'Confidence': eye_confidence,
                'Class Probabilities': dict(zip(eye_class_names, eye_probabilities))
            }
        else:
            # If predicted 'eye', use VGG model for further prediction
            prediction, confidence, probabilities = get_vgg_prediction(img_bytes)
                
            confidence_threshold = 0.6
            if confidence <= confidence_threshold:
                result = {
                    'Image': file.filename,
                    'Predicted Class': 'Unknown',
                    'Confidence': confidence,
                    'Class Probabilities': dict(zip(vgg_class_names, probabilities))
                }
            else:
                result = {
                    'Image': file.filename,
                    'Predicted Class': prediction,
                    'Confidence': confidence,
                    'Class Probabilities': dict(zip(vgg_class_names, probabilities))
                }

        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def index():
    return {"message": "Welcome to Eye Disease API! Use /predict for predictions."}

# Run the application with Uvicorn if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
