import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO)

def engineer_features(filepath, output_path):
    try:
        # Read the cleaned data
        df = pd.read_csv(filepath)

        # Feature Engineering: OneHotEncoding for categorical features
        categorical_features = df.select_dtypes(include=['object']).columns
        one_hot_encoder = OneHotEncoder(drop='first')
        one_hot_df = one_hot_encoder.fit_transform(df[categorical_features]).toarray()
        one_hot_df = pd.DataFrame(one_hot_df, columns=one_hot_encoder.get_feature_names_out(categorical_features))
        
        # Drop the original categorical columns
        df.drop(categorical_features, axis=1, inplace=True)
        
        # Concatenate original df with one_hot_df
        df = pd.concat([df, one_hot_df], axis=1)
        
        # Normalize Numerical Columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Save the processed DataFrame
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")

if __name__ == "__main__":
    input_path = r'student-ai-ml-tool\data\cleaned_data.csv'  # Replace with your cleaned data file path
    output_path = r'student-ai-ml-tool\data\processed_data.csv'  # Replace with your desired output file path
    engineer_features(input_path, output_path)
