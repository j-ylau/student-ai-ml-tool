import pickle
import pandas as pd
import logging
from utilities import setup_logger, load_config

setup_logger()

def make_prediction(input_data):
    try:
        with open('../model.pkl', 'rb') as f:
            model = pickle.load(f)
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return None
