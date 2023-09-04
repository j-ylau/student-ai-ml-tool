import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

def clean_data(filepath):
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return

        df = pd.read_csv(filepath)

        # Handle Missing Values
        for column in df.columns:
            if df[column].dtype == "object":
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)

        # Handle Outliers
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

        output_path = r'student-ai-ml-tool\data\cleaned_data.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")

if __name__ == "__main__":
    clean_data(r'student-ai-ml-tool\data\raw_data.csv')
