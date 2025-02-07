from flask import Flask, request, render_template
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        img = cv2.resize(img, (150, 150))
        img = img.reshape(1, 150, 150, 1)
        img = img / 255.0
        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            file_path = "static/uploads/" + file.filename
            file.save(file_path)
            
            img = preprocess_image(file_path)
            
            prediction = model.predict(img)
            probability = prediction[0][0]
            
            prediction_label = 'Pneumonia Positive' if probability <= 0.4 else 'Pneumonia Negative'
            
            return render_template('index.html', prediction=prediction_label, image=file_path, probability=probability)
        
        except ValueError as ve:
            error_message = str(ve)
            return render_template('index.html', error_message=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
