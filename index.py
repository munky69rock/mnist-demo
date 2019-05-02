from PIL import Image

import better_exceptions

from flask import Flask, jsonify, render_template, request

from mnist import Classifier

import numpy as np

app = Flask(__name__)
classifier = Classifier()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    image = Image.open(f)
    image.thumbnail((28, 28))
    # image.save("num.png")
    image = image.convert('L')
    image = 1.0 - np.asarray(image, dtype="float32") / 255
    image = image.reshape((1, 784))

    prediction = classifier.predict(image)
    return jsonify({str(k): float(v * 100) for k, v in prediction.items()})
