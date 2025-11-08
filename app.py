from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Label mapping
LABEL_MAP = {
    "pothole": "Road Department",
    "garbage": "Garbage Department",
    "streetlight": "Electricity Department",
    "water_leak": "Water Department",
    "fallen_tree": "Forest Department",
    "unknown": "General Department"
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    result = model(image)
    data = result.pandas().xyxy[0]

    if len(data) == 0:
        category = "unknown"
    else:
        category = data.iloc[0]['name']

    department = LABEL_MAP.get(category, "General Department")

    return jsonify({
        "category": category,
        "department": department
    })

@app.route('/', methods=['GET'])
def home():
    return "✅ CivicSense AI API is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
