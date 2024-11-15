import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import tree, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score



def plot_clf_tree(feature_names, class_names, clf, figsize: tuple = (14,10)):
  fig = plt.figure(figsize=figsize)
  plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True
  )
    
  plt.show()


def plot_clf_training_impurity_alpha(ccp_alphas, impurities, marker='o', drawstyle="steps-post"):
  fig, ax = plt.subplots()
  ax.plot(ccp_alphas[:-1], impurities[:-1], marker, drawstyle)
  ax.set_xlabel('effective alpha')
  ax.set_ylabel('total impurity of leaves')
  ax.set_title('Total Impurity vs effective alpha for training set')


def run_tree_classifier_on_dataset(df: pd.DataFrame, tree_clf, target_col: str, clf_params: dict = {}, param_grid: dict = {}, plot_alphas: bool = False, is_multiclass: bool = False):
  random_state = 12345
  average = 'weighted' if is_multiclass else 'binary'
  X = df.drop(target_col, axis=1)
  y = df[target_col]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

  if plot_alphas:
      # Get the cost complexity pruning path
      clf = tree_clf(random_state=random_state)
      path = clf.cost_complexity_pruning_path(X_train, y_train)
      ccp_alphas, impurities = path.ccp_alphas, path.impurities
      plot_clf_training_impurity_alpha(ccp_alphas, impurities)

      print(f"Min Alpha: {ccp_alphas[0]}\tMax Alpha: {ccp_alphas[-1]}")
      print(f"Num of alphas generated: {len(ccp_alphas)}")
      # Reducing num of alphas to 20 evenly
      ccp_alphas = np.linspace(np.min(ccp_alphas), np.max(ccp_alphas), 20)
      print(f"Num of alphas after reducing evenly: {len(ccp_alphas)}\n")

      clfs = []
      tree_num = 0
      for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

        if tree_num % 5 == 0:
            print(f"Tree: {tree_num}\t Number of nodes in the last tree is: {clfs[-1].tree_.node_count} with ccp_alpha: {ccp_alpha}") 
        tree_num += 1

      # plot the tree alphas
      node_counts = [clf.tree_.node_count for clf in clfs]
      depth       = [clf.tree_.max_depth  for clf in clfs]
      fig, ax     = plt.subplots(2, 1)

      # Num nodes vs alpha
      ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle='steps-post')
      ax[0].set_xlabel('alpha')
      ax[0].set_ylabel('number of nodes')
      ax[0].set_title('number of nodes vs alpha')

      # depth vs alpha
      ax[1].plot(ccp_alphas, depth, marker='o', drawstyle='steps-post')
      ax[1].set_xlabel('alpha')
      ax[1].set_ylabel('depth of tree')
      ax[1].set_title('depth vs alpha')      
      fig.tight_layout()

      # plot alpha an accuracy
      train_scores = [clf.score(X_train, y_train) for clf in clfs]
      test_scores  = [clf.score(X_test, y_test)   for clf in clfs]

      fig, ax = plt.subplots()
      ax.set_xlabel('alpha')
      ax.set_ylabel('accuracy')
      ax.set_title('Accuracy vs alpha for training and testing sets')
      ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
      ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
      ax.legend()
      plt.show()

  if param_grid:  
      clf = GridSearchCV(estimator=tree_clf(), param_grid=param_grid)
      clf.fit(X_train, y_train)
      clf = clf.best_estimator_  
      
  else:
      clf = tree_clf(random_state=random_state)
      clf.set_params(**clf_params)
      # Fit the clf
      clf.fit(X_train, y_train)

  y_train_pred = clf.predict(X_train)
  y_pred       = clf.predict(X_test)

  train_conf_matrix = confusion_matrix(y_train, y_train_pred)
  train_f1          = f1_score(y_train, y_train_pred, average=average)
  train_accuracy    = accuracy_score(y_train, y_train_pred)
  train_recall      = recall_score(y_train, y_train_pred, average=average)
  train_precision   = precision_score(y_train, y_train_pred, average=average)

  class_names = [str(cls) for cls in np.unique(y_train)]
  plot_clf_tree(X_train.columns, class_names, clf)

  # Predict the target values for the test set
  y_pred = clf.predict(X_test)
    
  # Calc resutls
  conf_matrix = confusion_matrix(y_test, y_pred)
  f1          = f1_score(y_test, y_pred, average=average)
  accuracy    = accuracy_score(y_test, y_pred)
  recall      = recall_score(y_test, y_pred, average=average)
  precision   = precision_score(y_test, y_pred, average=average)


  feature_importance = clf.feature_importances_
 # Get feature importance
  feature_importance = pd.DataFrame({
      'Feature': X.columns,
      'Importance': feature_importance
  }).sort_values(by='Importance', ascending=False)

  # Get the most significant features as a list of tuples
  feature_importance = list(feature_importance.itertuples(index=False, name=None))
  
  if is_multiclass:
      y_pred_proba = clf.predict_proba(X_test)
      roc_auc = {}
      optimal_thresholds_and_F1_scores = {}
      for i in range(len(clf.classes_)):
          y_test_binary = (y_test == clf.classes_[i]).astype(int)
          fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba[:, i])
          auc_score = auc(fpr, tpr)
          roc_auc[clf.classes_[i]] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

          f1_scores = []
          for threshold in thresholds:
              y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
              f1_scores.append(f1_score(y_test_binary, y_pred_binary))
          optimal_index = np.argmax(f1_scores)
          optimal_thresholds_and_F1_scores[clf.classes_[i]] = {
              'threshold': thresholds[optimal_index],
              'f1_score': f1_scores[optimal_index]
          }
  else:
      y_pred_proba = clf.predict_proba(X_test)[:, 1]
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
      auc_score = auc(fpr, tpr)
      roc_auc = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

      f1_scores = []
      for threshold in thresholds:
          y_pred_binary = (y_pred_proba >= threshold).astype(int)
          f1_scores.append(f1_score(y_test, y_pred_binary))
      optimal_index = np.argmax(f1_scores)
      optimal_thresholds_and_F1_scores = {
          'threshold': thresholds[optimal_index],
          'f1_score': f1_scores[optimal_index]
      }
  
  res = {
      "num_rows_trained": len(X_train),
      "Confusion Matrix": conf_matrix,
      "F1 Score": f1,
      "Accuracy": accuracy,
      "Recall": recall,
      "Precision": precision,
      "feature_importance": feature_importance,
      "Train Confusion Matrix": train_conf_matrix,
      "Train F1 Score": train_f1,
      "Train Accuracy": train_accuracy,
      "Train Recall": train_recall,
      "Train Precision": train_precision,
      "tree_hyperparameters": clf.get_params(),
      "ROC AUC": roc_auc,
      "optimal_thresholds_and_F1_scores": optimal_thresholds_and_F1_scores
  }

  return res, clf


def run_tree_ensamble_on_dataset(df: pd.DataFrame, clf, target_col: str, clf_params: dict = {}, is_multiclass=False) -> tuple:
  random_state = 12345
  average = 'weighted' if is_multiclass else 'binary'
  X = df.drop(target_col, axis=1)
  y = df[target_col]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

  # Create classifier
  clf = clf(random_state=random_state)
  clf.set_params(**clf_params)
  clf.fit(X_train, y_train)

  y_train_pred = clf.predict(X_train)
  y_pred       = clf.predict(X_test)

  train_conf_matrix = confusion_matrix(y_train, y_train_pred)
  train_f1          = f1_score(y_train, y_train_pred, average=average)
  train_accuracy    = accuracy_score(y_train, y_train_pred)
  train_recall      = recall_score(y_train, y_train_pred, average=average)
  train_precision   = precision_score(y_train, y_train_pred, average=average)

  class_names = [str(cls) for cls in np.unique(y_train)]

  # Predict the target values for the test set
  y_pred = clf.predict(X_test)
    
  # Calc resutls
  conf_matrix = confusion_matrix(y_test, y_pred)
  f1          = f1_score(y_test, y_pred, average=average)
  accuracy    = accuracy_score(y_test, y_pred)
  recall      = recall_score(y_test, y_pred, average=average)
  precision   = precision_score(y_test, y_pred, average=average)
  
  # Get feature importance
  if hasattr(clf, 'feature_importances_'):
    feature_importance = clf.feature_importances_
  else:
    feature_importance = [0] * X.shape[1]

  feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
  }).sort_values(by='Importance', ascending=False)

  # Get the most significant features as a list of tuples
  feature_importance = list(feature_importance.itertuples(index=False, name=None))

  # Get class names
  if hasattr(clf, 'classes_'):
      classes = clf.classes_
  else:
      classes = np.unique(y)

  # Calculate ROC curves and optimal thresholds
  y_pred_proba = clf.predict_proba(X_test)
  roc_auc = {}
  optimal_thresholds_and_F1_scores = {}
  
  if len(class_names) > 2:
      for i, class_label in enumerate(classes):
          y_test_binary = (y_test == class_label).astype(int)
          fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba[:, i])
          auc_score = auc(fpr, tpr)
          roc_auc[str(class_label)] = {
              'fpr': fpr.tolist(),
              'tpr': tpr.tolist(),
              'auc': auc_score
          }
          
          # Calculate F1 scores for different thresholds
          f1_scores = []
          for threshold in thresholds:
              y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
              f1_scores.append(f1_score(y_test_binary, y_pred_binary))
          
          optimal_index = np.argmax(f1_scores)
          optimal_thresholds_and_F1_scores[str(class_label)] = {
              'threshold': float(thresholds[optimal_index]),
              'f1_score': float(f1_scores[optimal_index])
          }
  else:
      # Binary classification case
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
      auc_score = auc(fpr, tpr)
      roc_auc[class_names[1]] = {
          'fpr': fpr.tolist(),
          'tpr': tpr.tolist(),
          'auc': auc_score
      }
      
      f1_scores = []
      for threshold in thresholds:
          y_pred_binary = (y_pred_proba[:, 1] >= threshold).astype(int)
          f1_scores.append(f1_score(y_test, y_pred_binary))
      
      optimal_index = np.argmax(f1_scores)
      optimal_thresholds_and_F1_scores = {
          class_names[1]: {
              'threshold': float(thresholds[optimal_index]),
              'f1_score': float(f1_scores[optimal_index])
          }
      }  
  # class_names = [str(cls) for cls in classes]
  
  # if len(class_names) > 2:
  #     y_pred_proba = clf.predict_proba(X_test)
  #     roc_auc = {}
  #     optimal_thresholds_and_F1scores = {}
  #     for i, class_label in enumerate(classes):
  #         y_test_binary = (y_test == class_label).astype(int)
  #         fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba[:, i])
  #         auc_score = auc(fpr, tpr)
  #         roc_auc[str(class_label)] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

  #         f1_scores = []
  #         for threshold in thresholds:
  #             y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
  #             f1_scores.append(f1_score(y_test_binary, y_pred_binary))
  #         optimal_index = np.argmax(f1_scores)
  #         optimal_thresholds_and_F1scores[str(class_label)] = {
  #             'threshold': thresholds[optimal_index],
  #             'f1_score': f1_scores[optimal_index]
  #         }
  # else:
  #     y_pred_proba = clf.predict_proba(X_test)[:, 1]
  #     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
  #     auc_score = auc(fpr, tpr)
  #     roc_auc = {class_names[0]: {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}}

  #     f1_scores = []
  #     for threshold in thresholds:
  #         y_pred_binary = (y_pred_proba >= threshold).astype(int)
  #         f1_scores.append(f1_score(y_test, y_pred_binary)) 
  #     optimal_index = np.argmax(f1_scores)
  #     optimal_thresholds_and_F1_scores = {
  #         'threshold': thresholds[optimal_index],
  #         'f1_score': f1_scores[optimal_index]
  #     }

  res = {
      "num_rows_trained": len(X_train),
      "Confusion Matrix": conf_matrix,
      "F1 Score": f1,
      "Accuracy": accuracy,
      "Recall": recall,
      "Precision": precision,
      "feature_importance": feature_importance,
      "Train Confusion Matrix": train_conf_matrix,
      "Train F1 Score": train_f1,
      "Train Accuracy": train_accuracy,
      "Train Recall": train_recall,
      "Train Precision": train_precision,
      "tree_hyperparameters": clf.get_params(),
      "ROC AUC": roc_auc,
      "optimal_thresholds_and_F1_scores": optimal_thresholds_and_F1_scores 
  }

  return res, clf


# Map the model feature back to the original features (pre encoding)
def plot_feature_importance(model_results_df: pd.DataFrame, model: str, model_cols: list, method: str):
    
    feature_importance_dict = {feature: [] for feature in model_cols}
    
    for org_feature in feature_importance_dict:
      for feature in model_results_df.loc[model]['feature_importance']:
        if org_feature in feature[0]:
          feature_importance_dict[org_feature].append(feature[1])
          
    if method == 'average':
      feature_importance_dict = {feature: np.average(val) for feature, val in feature_importance_dict.items()}
    else:
      feature_importance_dict = {feature: np.sum(val) for feature, val in feature_importance_dict.items()}      
    
    # Convert the list of tuples to a DataFrame
    df_top_features = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance']).sort_values(by="Importance", ascending=False)
    
    # Plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.barh(df_top_features['Feature'], df_top_features['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()