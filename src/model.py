import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import logging
from utilities import setup_logger, load_config

setup_logger()

def train_model(filepath):
    try:
        df = pd.read_csv(filepath)
        X = df.drop('Target', axis=1)
        y = df['Target']

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt'],
        }

        model = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        with open('../model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

    except Exception as e:
        logging.error(f"Error in model training: {e}")

if __name__ == "__main__":
    config = load_config()
    train_model(config['data']['processed_data_path'])
