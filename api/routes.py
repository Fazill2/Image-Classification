from flask import Blueprint, jsonify

from data_loading.data_loader import DataLoader

api_bp = Blueprint('api', __name__)

@api_bp.route('/train', methods=['POST'])
def train():
    try:
        data_loader = DataLoader('data')
        result = data_loader.train_df.size
        return jsonify({'message': 'Training completed', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/test', methods=['POST'])
def test():
    try:
        data_loader = DataLoader('data')
        result = data_loader.test_df.size
        return jsonify({'message': 'Testing completed', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
