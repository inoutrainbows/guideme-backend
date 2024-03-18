from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import ssl
import certifi
import base64

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Initialize Flask application and the pre-trained model
app = Flask(__name__)
model = MobileNetV2(weights='imagenet')

def prepare_image(img):
    """
    Preprocesses the image for MobileNetV2.
    """
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image_data' in request.form:
            # Decode the base64-encoded image
            encoded_image = request.form['image_data']
            decoded_image = base64.b64decode(encoded_image)
            img = Image.open(io.BytesIO(decoded_image))
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            img = Image.open(io.BytesIO(file.read()))
        else:
            return jsonify({"error": "No image data found"}), 400

        # Read the image and preprocess it
        processed_image = prepare_image(img)

        # Make prediction
        preds = model.predict(processed_image)
        results = decode_predictions(preds, top=3)[0]  # Top 3 predictions

        # Format the results as a list of JSON objects
        response = [{"label": result[1], "confidence": float(result[2])} for result in results]
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction: " + str(e)}), 500

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
