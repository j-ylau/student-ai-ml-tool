import pandas as pd
import logging
from utilities import setup_logger, load_config
from flask import Flask, request, jsonify
from src.prediction import make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = make_prediction(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

setup_logger()

def clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Your complex data cleaning logic here
        # For example, imputation, outlier removal, etc.
        df.dropna(inplace=True)
        df.to_csv('../data/cleaned_data.csv', index=False)
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")

if __name__ == "__main__":
    config = load_config()
    clean_data(config['data']['raw_data_path'])
