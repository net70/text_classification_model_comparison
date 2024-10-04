from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np

import json
import re


def read_pandas_csv_clean_columns_names(csv_file_path: str) -> pd.DataFrame:
    def clean_column_name(col: str) -> str:
        # Check if the column name is already in snake_case using regex, else clean it
        return col if re.match(r'^[a-z0-9_]+$', col) else re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z0-9\s]', '', col.strip())).lower()

    df = pd.read_csv(csv_file_path)
    df.columns = [clean_column_name(col) for col in df.columns]
    return df
    

def get_model_cross_validation(modef_df: pd.DataFrame, target_col: str, prediction_col: str) -> dict:
# Extract the actual and predicted labels
    y_true = modef_df[target_col]
    y_pred = modef_df[prediction_col]
    
    # Calculate metrics
    results = {
        'accuracy':           accuracy_score(y_true, y_pred),
        'f1_macro':           f1_score(y_true, y_pred, average='macro'),
        'f1_micro':           f1_score(y_true, y_pred, average='micro'),
        'f1_weighted':        f1_score(y_true, y_pred, average='weighted'),
        'precision_macro':    precision_score(y_true, y_pred, average='macro'),
        'precision_micro':    precision_score(y_true, y_pred, average='micro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro':       recall_score(y_true, y_pred, average='macro'),
        'recall_micro':       recall_score(y_true, y_pred, average='micro'),
        'recall_weighted':    recall_score(y_true, y_pred, average='weighted')
    }
    
    # Return the results dictionary
    return results