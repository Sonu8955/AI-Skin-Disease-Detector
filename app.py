from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('skin_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes = ['acne', 'eczema', 'psoriasis', 'ringworm']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    if file:

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filepath)

        img = image.load_img(filepath, target_size=(128,128))
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array /= 255.0

        prediction = model.predict(img_array)

        result = classes[np.argmax(prediction)]

        confidence = round(100 * np.max(prediction), 2)

        return render_template(
            'result.html',
            prediction=result,
            confidence=confidence,
            image_path=filepath
        )

    return "No File Uploaded"

if __name__ == '__main__':
    app.run(debug=True)