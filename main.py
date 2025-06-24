import io
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision import models,transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image,ImageDraw
import os

# Load your custom trained model
def load_custom_model():
    # Create the same model architecture as your training
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Update the classifier head for your number of classes (2 classes: background + gun)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load your trained weights
    model_path = "artifacts/models/fasterrcnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"Model file not found at {model_path}, using untrained model")
    
    return model

model = load_custom_model()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])


app = FastAPI()

def predict_and_draw(image : Image.Image):
    
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # Class names for your gun detection model
    class_names = {0: 'background', 1: 'gun'}
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Lower threshold since it's your trained model
            x_min, y_min, x_max, y_max = box
            
            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
            # Draw label and confidence score
            class_name = class_names.get(label, 'unknown')
            text = f"{class_name}: {score:.2f}"
            draw.text((x_min, y_min-20), text, fill="red")
    
    return img_rgb


@app.get("/")
def read_root():
    return {"message" : "Welcome to the Guns Object Detection API"}


@app.post("/predict/")
async def predict(file:UploadFile=File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    output_image = predict_and_draw(image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr , format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr , media_type="image/png")