from flask import Flask, jsonify, request
from utils import pred_users
import numpy as np
app = Flask(__name__)

"""
Prediction Route.
body:
[
    [user_id, content_id, content_type_id, timestamp],
    ...
]
"""
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            preds = pred_users(np.array(request.json))

            return jsonify({
                "preds": preds
            })
        except:
            return jsonify({
                "error": "Error occured during prediction."
            })
