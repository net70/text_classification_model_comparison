import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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


def run_svm_on_dataset(df: pd.DataFrame, target_col: str, clf_params: dict = {}):
  random_state = 12345
  X = df.drop(target_col, axis=1)
  y = df[target_col]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

  svm = SVC(random_state=random_state)
  svm.set_params(**clf_params)
  svm.fit(X_train, y_train)

  plot_svm(X_train, X_test, y_train, y_test, clf_params)
    
  y_train_pred = svm.predict(X_train)
  y_pred       = svm.predict(X_test)

  train_conf_matrix = confusion_matrix(y_train, y_train_pred)
  train_f1          = f1_score(y_train, y_train_pred)
  train_accuracy    = accuracy_score(y_train, y_train_pred)
  train_recall      = recall_score(y_train, y_train_pred)
  train_precision   = precision_score(y_train, y_train_pred)

  class_names = np.unique(label_encoder.inverse_transform(y_train))

  # Predict the target values for the test set
  y_pred = svm.predict(X_test)
    
  # Calc resutls
  conf_matrix = confusion_matrix(y_test, y_pred)
  f1          = f1_score(y_test, y_pred)
  accuracy    = accuracy_score(y_test, y_pred)
  recall      = recall_score(y_test, y_pred)
  precision   = precision_score(y_test, y_pred)


  res = {
      "num_rows_trained": len(X_train),
      "Confusion Matrix": conf_matrix,
      "F1 Score": f1,
      "Accuracy": accuracy,
      "Recall": recall,
      "Precision": precision,
      "feature_importance": None,
      "Train Confusion Matrix": train_conf_matrix,
      "Train F1 Score": train_f1,
      "Train Accuracy": train_accuracy,
      "Train Recall": train_recall,
      "Train Precision": train_precision,
      "KNN_K": None,
      "test_Coefficients": None,
      "test_Intercept": None,
      "tree_hyperparameters": None,
      "svm_hyperparameters": svm.get_params()
  }

  return res