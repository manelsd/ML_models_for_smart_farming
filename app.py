import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
with open('randomforest.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    float_features = [float(x) for x in request_data.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return jsonify({'prediction_text': 'soil might be damaged' if prediction[0] == 0 else 'your soil is in good health'})

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

