from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
import numpy as np
from keras.models import load_model
import os
import logging
from pymongo import MongoClient

# Create a MongoDB client
client = MongoClient("mongodb://localhost:27017/")  

# Create a database and collection
db = client["predictions_database"]  # Replace with your database name
collection = db["predictions_collection"]


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'uploads')
STATIC_FOLDER = os.path.join(dir_path, 'static')

app = Flask(__name__)
model = load_model(os.path.join(STATIC_FOLDER, 'mnist_cnn.h5'))

# Configure logging
log_file = os.path.join(dir_path, 'prediction.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['image']
    data = data.replace('data:image/png;base64,', '')
    data = data.replace(' ', '+') 
    imgdata = base64.b64decode(data)
    filename = os.path.join(UPLOAD_FOLDER, 'test.jpg')

    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    # Log the prediction request
    logging.info(f"Prediction requested for image: {filename}")

    result = predict(filename)
    print("Result:", result)
    predicted_number = result.argmax(axis=1).item()
    print("Predicted number", predicted_number)    

    # Log the prediction result
    logging.info(f"Prediction result: {predicted_number}")

    # Save the prediction result to MongoDB    
    new_prediction = {
        'filename': filename,
        'predicted_number': predicted_number
    }
    collection.insert_one(new_prediction)
    print("Inserted into the mongodb database")

    return jsonify({'label': predicted_number})

def predict(filename):
    img = Image.open(filename)
    img = img.resize((28, 28))
    img = np.asarray(img)
    data = img[:, :, 3]  # select only the visible channel
    data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=0)

    predicted = model.predict(data)
    return predicted

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
