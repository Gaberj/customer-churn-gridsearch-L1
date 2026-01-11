# gridsearch_model.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_gridsearch_model(X_train, y_train, param_grid):
    """
    Train a logistic regression model using GridSearchCV.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        param_grid (dict): Grid search parameters.

    Returns:
        best_model (LogisticRegression): Best model from GridSearchCV.
        grid_search (GridSearchCV): The GridSearchCV object.
    """
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model's performance on test data.

    Args:
        model (LogisticRegression): Trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.

    Returns:
        metrics (dict): Dictionary containing accuracy, AUC-ROC, and classification report.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Best Model)')
    plt.show()

    return {'accuracy': accuracy, 'auc_roc': auc_roc, 'classification_report': report}
