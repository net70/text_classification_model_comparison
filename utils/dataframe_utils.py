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
    

def extract_nested_str_lst_features(df: pd.DataFrame, str_lst_col: str, prefix: str) -> pd.DataFrame:
    # Get all unique values
    values = df[str_lst_col].explode().unique()
    df[str_lst_col] = df[str_lst_col].apply(lambda x: set(x))

    # Create features for each pos type
    for value in values:
       df[f'{prefix}_{value}'] = df[str_lst_col].apply(lambda x: 1 if value in x else 0)
    df = df.drop(str_lst_col, axis=1)

    return df


def extract_named_entities_to_columns(df: pd.DataFrame, ner_col: str) -> pd.DataFrame:
    # Get all unique entity types
    entity_types = set(ent['type'] for sublist in df[ner_col] for ent in sublist)
    
    # Create features for each entity type
    for entity_type in entity_types:
        df[f'entity_{entity_type}'] = df[ner_col].apply(
            lambda ents: sum(1 for ent in ents if ent['type'] == entity_type)
        )
    df = df.drop(ner_col, axis=1)

    return df
    

def embeddings_to_columns(df: pd.DataFrame, embedding_col: str) -> pd.DataFrame:
    embeddings_df = pd.DataFrame(df[embedding_col].tolist())
    #embeddings_df.columns = [f'{embedding_col}_{i}' for i in range(embeddings_df.shape[1])]
    embeddings_df.columns = [f'col_embedding_{i+1}' for i in embeddings_df.columns]

    df_expanded = pd.concat([df.drop(embedding_col, axis=1).reset_index(drop=True), embeddings_df], axis=1)
    # df = pd.concat([df, embeddings_df], axis=1)
    # df.drop(embedding_col, axis=1, inplace=True)
    
    return df_expanded


def remove_outliers(df: pd.DataFrame, columns: list, rows_to_keep_col: str = None):
  initial_shape = df.shape[0]
  for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if rows_to_keep_col:
        df = df[((df[column] >= lower_bound) & (df[column] <= upper_bound)) | (df[rows_to_keep_col]==True)]    
    else:
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removing outliers in {column}: {initial_shape - df.shape[0]} rows removed.")
    initial_shape = df.shape[0]

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
    
    return results
