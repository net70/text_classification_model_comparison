import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import json
import re


def read_pandas_csv_clean_columns_names(csv_file_path: str) -> pd.DataFrame:
    def clean_column_name(col: str) -> str:
        return col if re.match(r'^[a-z0-9_]+$', col) else re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z0-9\s]', '', col.strip())).lower()

    df = pd.read_csv(csv_file_path)
    df.columns = [clean_column_name(col) for col in df.columns]
    return df
    

def extract_nested_str_lst_features(df: pd.DataFrame, str_lst_col: str, prefix: str) -> pd.DataFrame:
    values = df[str_lst_col].explode().unique()
    df[str_lst_col] = df[str_lst_col].apply(lambda x: set(x))

    for value in values:
       df[f'{prefix}_{value}'] = df[str_lst_col].apply(lambda x: 1 if value in x else 0)
    df = df.drop(str_lst_col, axis=1)

    return df


def extract_named_entities_to_columns(df: pd.DataFrame, ner_col: str) -> pd.DataFrame:
    entity_types = set(ent['type'] for sublist in df[ner_col] for ent in sublist)
    
    for entity_type in entity_types:
        df[f'entity_{entity_type}'] = df[ner_col].apply(
            lambda ents: sum(1 for ent in ents if ent['type'] == entity_type)
        )
    df = df.drop(ner_col, axis=1)

    return df


def remove_outliers(df: pd.DataFrame, columns: list, rows_to_keep_col: str = None) -> pd.DataFrame:
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


def embeddings_to_columns(df: pd.DataFrame, embedding_col: str) -> pd.DataFrame:
    embeddings_df = pd.DataFrame(df[embedding_col].tolist())
    embeddings_df.columns = [f'col_embedding_{i+1}' for i in embeddings_df.columns]

    df_expanded = pd.concat([df.drop(embedding_col, axis=1).reset_index(drop=True), embeddings_df], axis=1)
    
    return df_expanded


def get_balanced_dataset(original_df: pd.DataFrame, classes_col: str, synthetic_col='is_synthetic', random_state=12345) -> pd.DataFrame:
    X = original_df.drop(classes_col, axis=1)
    y = original_df[classes_col]

    majority_class = original_df[classes_col].value_counts().nlargest(1).index[0]
    majority_class_count = original_df[classes_col].value_counts().nlargest(1)[1]

    desired_samples = {
        2: majority_class_count,
        0: majority_class_count
    }
    
    # Apply SMOTE to generate synthetic samples
    smote = SMOTE(sampling_strategy=desired_samples, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    synthetic_samples_df = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_samples_df[synthetic_col] = True
    synthetic_samples_df[classes_col] = y_resampled
    
    max_index = original_df.index.max()
    synthetic_samples_df.index = range(max_index + 1, max_index + 1 + len(synthetic_samples_df))
  
    original_df[synthetic_col] = False
    balanced_df = pd.concat([original_df.reset_index(drop=True), synthetic_samples_df])
    
    return balanced_df

def get_sales_data_set_optimzation_metrics(df: pd.DataFrame, target_col: str, prediction_col: str):
  print("##### DATA SET BASE OPTIMIZATION FUNCTION #####")
  num_qualified = len(df[df['status'] == 'Qualified'])
  num_sales_qualified      = len(df[(df['status'] == 'Qualified') & (df[target_col] == 'sales')])
  num_support_qualified    = len(df[(df['status'] == 'Qualified') & (df[target_col] == 'support')])
  num_irrelevant_qualified = len(df[(df['status'] == 'Qualified') & (df[target_col] == 'irrelevant')])
  
  print(f"percent sales & Qualified: {round(num_sales_qualified/num_qualified*100, 1)}")
  print(f"percent support & Qualified: {round(num_support_qualified/num_qualified*100, 1)}")
  print(f"percent irrelevant & Qualified: {round(num_irrelevant_qualified/num_qualified*100, 1)}")
  
  num_unqualified = len(df[df['status'] == 'Unqualified'])
  num_sales_unqualified      = len(df[(df['status'] == 'Unqualified') & (df[target_col] == 'sales')])
  num_support_unqualified    = len(df[(df['status'] == 'Unqualified') & (df[target_col] == 'support')])
  num_irrelevant_unqualified = len(df[(df['status'] == 'Unqualified') & (df[target_col] == 'irrelevant')])
  print("")
  print(f"percent sales & Unqualified: {round(num_sales_unqualified/num_unqualified*100, 1)}")
  print(f"percent support & Unqualified: {round(num_support_unqualified/num_unqualified*100, 1)}")
  print(f"percent irrelevant & Unqualified: {round(num_irrelevant_unqualified/num_unqualified*100, 1)}")  

  print("\n\n##### PREDICTION OPTIMIZATION FUNCTION #####")
  num_qualified = len(df[df['status'] == 'Qualified'])
  num_sales_qualified      = len(df[(df['status'] == 'Qualified') & (df[prediction_col] == 'sales')])
  num_support_qualified    = len(df[(df['status'] == 'Qualified') & (df[prediction_col] == 'support')])
  num_irrelevant_qualified = len(df[(df['status'] == 'Qualified') & (df[prediction_col] == 'irrelevant')])
  
  print(f"percent sales & Qualified: {round(num_sales_qualified/num_qualified*100, 1)}")
  print(f"percent support & Qualified: {round(num_support_qualified/num_qualified*100, 1)}")
  print(f"percent irrelevant & Qualified: {round(num_irrelevant_qualified/num_qualified*100, 1)}")
  
  num_unqualified = len(df[df['status'] == 'Unqualified'])
  num_sales_unqualified      = len(df[(df['status'] == 'Unqualified') & (df[prediction_col] == 'sales')])
  num_support_unqualified    = len(df[(df['status'] == 'Unqualified') & (df[prediction_col] == 'support')])
  num_irrelevant_unqualified = len(df[(df['status'] == 'Unqualified') & (df[prediction_col] == 'irrelevant')])
  print("")
  print(f"percent sales & Unqualified: {round(num_sales_unqualified/num_unqualified*100, 1)}")
  print(f"percent support & Unqualified: {round(num_support_unqualified/num_unqualified*100, 1)}")
  print(f"percent irrelevant & Unqualified: {round(num_irrelevant_unqualified/num_unqualified*100, 1)}")    

def get_validation_dataframe(df: pd.DataFrame, target_col: str, sample_size: int, random_state=12345) -> tuple:

    df_validations = []
    classes = df[target_col].unique()
    for cls in classes:
      df_class = df[df[target_col] == cls].sample(sample_size // len(classes), random_state=random_state)
      df_validations.append(df_class)
    
    df_validation = pd.concat(df_validations, ignore_index=True)
    df_train_test = df.drop(df_validation.index)
    
    return df_validation, df_train_test  


def get_train_test_sets(df: pd.DataFrame, target_col: str, test_ratio=0.2, random_state=12345) -> tuple:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def get_model_cross_validation(model_df: pd.DataFrame, target_col: str, prediction_col: str) -> dict:
    y_true  = model_df[target_col]
    y_pred  = model_df[prediction_col]
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
