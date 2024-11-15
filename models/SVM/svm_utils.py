import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

random_state = 12345

def plot_svm(X_train, X_test, y_train, y_test, clf_params):
    pca         = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    svm = SVC(random_state=12345)
    svm.set_params(**clf_params)
    svm.fit(X_train_pca, y_train)    
    
    X_combined_std = np.vstack((X_train_pca, X_test_pca))
    y_combined     = np.hstack((y_train, y_test))
    
    plot_decision_regions(X_combined_std, y_combined, clf=svm)

    plt.xlabel('X_combined_std [standardized]')
    plt.ylabel('Ycombined [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def find_optimal_threshold(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, f1_scores[optimal_idx]


def run_svm_on_dataset(df: pd.DataFrame, target_col: str, clf_params: dict = {}, is_binary=True):
    avg = 'binary' if is_binary else 'macro'
    random_state = 12345
    class_names = np.unique(df[target_col])
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    if is_binary:
        svm = SVC(random_state=random_state)
        svm.set_params(**clf_params)
    else:
        svm = OneVsRestClassifier(SVC(random_state=random_state, **clf_params))

    svm.fit(X_train, y_train)
    plot_svm(X_train, X_test, y_train, y_test, clf_params)

    y_train_pred = svm.predict(X_train)
    y_pred = svm.predict(X_test)
  
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average=avg)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred, average=avg)
    train_precision = precision_score(y_train, y_train_pred, average=avg)

    # Get decision function scores for multiclass
    y_score = svm.decision_function(X_test)

    if is_binary:
        # Binary classification: optimize threshold and compute metrics
        optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_score)
        print(f"Optimal Threshold: {optimal_threshold}, Best F1 Score: {best_f1}")

        # Apply optimal threshold
        y_pred_custom_threshold = (y_score >= optimal_threshold).astype(int)

        # Calculate ROC AUC with the optimal threshold
        fpr, tpr, _ = roc_curve(y_test, y_pred_custom_threshold)
        roc_auc = {'binary': {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}}
        best_thresholds_and_f1s = {'binary': {'threshold': optimal_threshold, 'f1_score': best_f1}}

    else:
        # Multiclass classification: optimize threshold for each class
        best_thresholds_and_f1s = {}
        y_pred_custom_threshold = np.zeros_like(y_test)

        for i in range(y_score.shape[1]):
            y_test_binary = (y_test == i).astype(int)
            optimal_threshold, best_f1 = find_optimal_threshold(y_test_binary, y_score[:, i])

            # Store the optimal threshold and best F1 score for this class
            best_thresholds_and_f1s[i] = {'threshold': optimal_threshold, 'f1_score': best_f1}

            # Apply the optimal threshold for each class
            y_pred_class = (y_score[:, i] >= optimal_threshold).astype(int)
            y_pred_custom_threshold[y_pred_class == 1] = i  # Assign class if threshold is met

        # Calculate ROC AUC using threshold-adjusted predictions
        roc_auc = {}
        for i in range(y_score.shape[1]):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_custom_threshold == i)
            roc_auc[i] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}

    # Calculate metrics with the custom threshold
    conf_matrix = confusion_matrix(y_test, y_pred_custom_threshold)
    f1 = f1_score(y_test, y_pred_custom_threshold, average=avg)
    accuracy = accuracy_score(y_test, y_pred_custom_threshold)
    recall = recall_score(y_test, y_pred_custom_threshold, average=avg)
    precision = precision_score(y_test, y_pred_custom_threshold, average=avg)

    # Collect results
    res = {
        "num_rows_trained": len(X_train),
        "Confusion Matrix": conf_matrix,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "Train Confusion Matrix": train_conf_matrix,
        "Train F1 Score": train_f1,
        "Train Accuracy": train_accuracy,
        "Train Recall": train_recall,
        "Train Precision": train_precision,
        "svm_hyperparameters": svm.get_params(),
        "ROC AUC": roc_auc,
        "optimal_thresholds_and_F1_scores": best_thresholds_and_f1s 
    }

    return res, svm


def predict_with_thresholds(model, X_new, optimal_thresholds, is_binary=True):
    # Get decision function scores for new data
    y_score_new = model.decision_function(X_new)
    
    if is_binary:
        # Apply the binary threshold
        optimal_threshold = optimal_thresholds['binary']['threshold']
        y_pred_new = (y_score_new >= optimal_threshold).astype(int)
    
    else:
        # Apply the thresholds for each class (multiclass)
        y_pred_new = np.zeros((X_new.shape[0],), dtype=int)
        for i in range(y_score_new.shape[1]):
            threshold = optimal_thresholds[i]['threshold']
            # Apply the threshold for class `i`
            y_pred_class = (y_score_new[:, i] >= threshold).astype(int)
            y_pred_new[y_pred_class == 1] = i  # Assign the class if threshold is met
    
    return y_pred_new


def get_permutation_importance(model, df: pd.DataFrame, target_col: str, n_repeats=10, random_state=12345, n_jobs=-1) -> pd.DataFrame:
  
  X = df.drop(target_col, axis=1)
  y = df[target_col]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

  # Compute permutation feature importance
  result = permutation_importance(
    model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
  )  

  # Store feature importance in a DataFrame
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance Mean': result.importances_mean,
      'Importance Std': result.importances_std
  })

  importance_df = importance_df.sort_values(by='Importance Mean', ascending=False)
  importance_df.reset_index(drop=True, inplace=True)

  return importance_df

  