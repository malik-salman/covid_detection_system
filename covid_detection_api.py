from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__) 
CORS(app)

# Load saved models
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data.columns = [col.replace('Fever', 'Fever (°C)') for col in input_data.columns]

        nb_pred = nb_model.predict(input_data)[0]
        svm_pred = svm_model.predict(input_data)[0]
        knn_pred = knn_model.predict(input_data)[0]

        votes = [nb_pred, svm_pred, knn_pred]
        final_pred = 1 if votes.count(1) > votes.count(0) else 0

        return jsonify({
            "Naive Bayes Prediction": int(nb_pred),
            "SVM Prediction": int(svm_pred),
            "KNN Prediction": int(knn_pred),
            "Final Prediction": int(final_pred)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)