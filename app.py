from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# load the trained model which is serialized
model = joblib.load('expense_model.joblib')

@app.route('/hello', methods=['POST'])
def hello():
    # check for JSON content type
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    return jsonify({"message": "Hello, World!", "data": data})


@app.route('/predict', methods=['POST'])
def predict():
    # get JSON request 
    data = request.get_json()

    # feature extraction from dataset
    age = data['age']
    bmi = data['bmi']
    children = data['children']
    sex = data['sex']  # 0 for female, 1 for male
    smoker = data['smoker']  # 0 for non-smoker, 1 for smoker
    region = data['region']  # encoded region value(0,1,2,3 for northwest, northeast, southwest, southeast)

    # feature array for prediction
    features = np.array([[age, bmi, children, sex, smoker, region]])

    # predict the medical charges
    prediction = model.predict(features)

    # return the prediction charges as a JSON response
    return jsonify({'predicted_charges': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
