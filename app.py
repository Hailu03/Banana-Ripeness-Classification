from flask import Flask, request, jsonify
import torch 
from torchvision import transforms 
from PIL import Image 
import io
import base64
from utils.Efficientnet import EfficientNet
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

NUM_OF_CLASSES = 4
width_mult, depth_mult, res, dropout_rate = [1.0, 1.0, 224, 0.2]
model = EfficientNet(width_mult, depth_mult, dropout_rate, num_classes=NUM_OF_CLASSES)
model.load_state_dict(torch.load('model/efficient_net.pth', map_location='cpu'))
model.eval()
labels_ripeness = ['Green','Midripen','Overripen','Yellowish_Green']

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    print("Connection Established")
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = transform_image(image_data)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return jsonify({'class': labels_ripeness[predicted.item()]})
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Hello World'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all interfaces

