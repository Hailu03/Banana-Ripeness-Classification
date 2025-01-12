from flask import Flask, render_template, Response, request, jsonify
import subprocess
import os
import threading
import time
from io import BytesIO
import torch
from torchvision import transforms
from PIL import Image
import base64
from utils.Efficientnet import EfficientNet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
FRAME_FILE = "current_frame.jpg"  # Frame file for live video
CAPTURE_FILE = "captured_frame.jpg"  # File to save the captured frame
TEMP_FILE = "temp_frame.jpg"

# EfficientNet model setup
#NUM_OF_CLASSES = 4
NUM_OF_CLASSES = 2
width_mult, depth_mult, res, dropout_rate = [1.0, 1.0, 224, 0.2]
model = EfficientNet(width_mult, depth_mult, dropout_rate, num_classes=NUM_OF_CLASSES)
model.load_state_dict(torch.load('model/efficient_net_3.pth', map_location='cpu'))
#model.load_state_dict(torch.load('model/efficient_net.pth', map_location='cpu'))
model.eval()
#labels_ripeness = ['Green', 'Midripen', 'Overripen', 'Yellowish_Green']
labels_ripeness = ['Riped',"Unriped"]

# Transform function for image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
    ])
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

# Function to capture video frames using libcamera-vid
def capture_video():
    while True:
        command = [
            "libcamera-vid",
            "--codec", "mjpeg", "--width", "640", "--height", "480", 
            "-t", "0", "--output", "-",  # Infinite stream to stdout
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

        try:
            while True:
                marker = process.stdout.read(2)
                if marker != b'\xff\xd8':  # JPEG start marker
                    continue

                frame = marker
                while True:
                    chunk = process.stdout.read(4096)
                    if b'\xff\xd9' in chunk:  # JPEG end marker
                        frame += chunk[:chunk.index(b'\xff\xd9') + 2]
                        break
                    frame += chunk

                # Save frame to temporary file, then atomically replace it
                with open(TEMP_FILE, "wb") as temp_f:
                    temp_f.write(frame)
                os.replace(TEMP_FILE, FRAME_FILE)

        except Exception as e:
            print(f"Error capturing video: {e}")
        finally:
            process.terminate()
            time.sleep(1)

# MJPEG generator for streaming video
def generate_frames():
    while True:
        try:
            with open(FRAME_FILE, "rb") as img_file:
                frame = img_file.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except FileNotFoundError:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\xFF\xD8\xFF\xD9' + b'\r\n')
        time.sleep(0.1)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    # Copy the latest frame to a "captured" file
    if os.path.exists(FRAME_FILE):
        os.system(f"cp {FRAME_FILE} {CAPTURE_FILE}")
        with open(CAPTURE_FILE, "rb") as img_file:
            img_data = img_file.read()
            image = transform_image(img_data)

        # Measure prediction time
        start_time = time.time()  # Start timer
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = labels_ripeness[predicted.item()]
        end_time = time.time()  # End timer

        prediction_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        print(f"Prediction Time: {prediction_time} ms")

        return jsonify({'class': result, 'prediction_time_ms': prediction_time})
    return jsonify({'error': 'No frame available to capture'}), 400

if __name__ == '__main__':
    # Start the video capture thread
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)

