from flask import Blueprint, jsonify, request

from data_loading.data_loader import DataLoader
from models.models import ModelHandler

api_bp = Blueprint('api', __name__)
model_handler = ModelHandler()

@api_bp.route('/model', methods=['GET'])
def model():
    try:
        return jsonify({'models': model_handler.model_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        model_name = data.get('model')
        data_loader = DataLoader('data')
        model_handler.train(model_name, data_loader.train_df, data_loader.train_y)
        return jsonify({'message': 'Training completed', 'result': "success"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/test', methods=['POST'])
def test():
    try:
        data = request.get_json()
        model_name = data.get('model')
        data_loader = DataLoader('data')
        scores = model_handler.test(model_name, data_loader.test_df, data_loader.test_y)
        return jsonify({'message': 'Testing completed', 'scores': scores}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data.get('model')
        prediction = model_handler.predict(model_name, {})
        return jsonify({'message': 'Prediction completed', 'result': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
