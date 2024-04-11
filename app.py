
from flask import Flask, request, jsonify  
from flask_cors import CORS 
import io  
import base64  
import logging 
from PIL import Image  
from openai import OpenAI 
import sqlite3
#from pdb import set_trace
#Connection to Databse

# Configure logging level and format
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# API key for OpenAI (this should be kept secret)
openai_api_key = ''  

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# Initialize the OpenAI client with the provided API key
client = OpenAI(api_key=openai_api_key)

# Function to decode image data from base64
def decode_image(image_data):
    logging.debug("Decoding base64 image data")
    return base64.b64encode(image_data).decode("utf-8")

@app.route('/save_contacts', methods=['POST'])
def save_contacts():
    data = request.json
    con = sqlite3.connect("guide_me_database.db") 
    cur = con.cursor()
    cur.execute(f"""
        INSERT INTO contacts VALUES
            ('{data['name']}','{data['number']}')      
    """)
    con.commit()
    return data


# Define a route for the /predict endpoint, accepting POST requests
@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request on /predict endpoint")
    try:
        # Get JSON data from the request
        data = request.json
        # Check if 'image_data' is not in the request
        if not data or 'image_data' not in data:
            logging.error("No image_data found in request")
            return jsonify({"error": "No image_data found in request"}), 400

        # Decode the base64 image data
        image_data = base64.b64decode(data['image_data'])
        # Open the image using PIL
        img = Image.open(io.BytesIO(image_data))
        # Encode the image data back to base64
        encoded_string = decode_image(image_data)

        # System prompt for OpenAI
        system_prompt = "You are an expert at analyzing images."
        # Call OpenAI's API with the prompt and image
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
                    "content": [{"type": "text", "text": "I'm a visually impaired person. Describe this image in a 5 or 6 words sentence. Reply and provide An Arabic Translation"}],
                },
            ],
            max_tokens=1000,
        )
        # Return the response from OpenAI
        return jsonify({"message": response.choices[0].message.content})

    except Exception as e:
        # Log and return the error if an exception occurs
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Start the Flask app if this file is run directly
if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=8080)
