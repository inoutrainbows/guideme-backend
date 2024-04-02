from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import logging
from PIL import Image
from openai import OpenAI

# Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
openai_api_key = ''
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=openai_api_key)

def decode_image(image_data):
    logging.debug("Decoding base64 image data")
    return base64.b64encode(image_data).decode("utf-8")

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request on /predict endpoint")
    try:
        data = request.json
        if not data or 'image_data' not in data:
            logging.error("No image_data found in request")
            return jsonify({"error": "No image_data found in request"}), 400

        image_data = base64.b64decode(data['image_data'])
        img = Image.open(io.BytesIO(image_data))
        encoded_string = decode_image(image_data)

        system_prompt = "You are an expert at analyzing images."
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this image for a blind person in only 5 words."}],
                },
            ],
            max_tokens=1000,
        )
        return jsonify({"message": response.choices[0].message.content})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=8080)
