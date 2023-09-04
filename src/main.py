from data_preprocessing import clean_data
from feature_engineering import engineer_features
from model import train_model
from prediction import make_prediction
from utilities import load_config

if __name__ == "__main__":
    config = load_config()
    clean_data(config['data']['raw_data_path'])
    engineer_features(config['data']['cleaned_data_path'])
    train_model(config['data']['processed_data_path'])
    sample_input = {
        'feature_1': 0.5,
        'feature_2': 0.3,
        'feature_3': 0.7,
    }
    print("Sample Prediction:", make_prediction(sample_input))
