from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)
model = ResNet50(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    return preprocess_input(img_array)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image_data' not in data:
            return jsonify({"error": "No image data found"}), 400

        # Decode base64 image
        decoded_image = base64.b64decode(data['image_data'])
        img = Image.open(io.BytesIO(decoded_image))

        # Preprocess image
        processed_image = preprocess_image(img)

        # Make predictions
        preds = model.predict(processed_image)
        predictions = decode_predictions(preds, top=3)[0]

        # Construct response
        response_data = [{"label": label, "confidence": float(confidence)} for _, label, confidence in predictions]
        message = "Top 3 Predictions:\n" + "\n".join([f"{i + 1}: {pred['label']} with confidence {pred['confidence']:.2f}" for i, pred in enumerate(response_data)])
        response = {"message": message, "predictions": response_data}
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction: " + str(e)}), 500


    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
