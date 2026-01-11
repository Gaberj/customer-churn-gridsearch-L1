# lasso_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_lasso_model(X_train, y_train, C=0.1):
    """
    Train an L1-regularized logistic regression model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        C (float): Regularization strength (default: 0.1).

    Returns:
        model (LogisticRegression): Trained logistic regression model.
    """
    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


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
    plt.title('Confusion Matrix (L1 Regularization)')
    plt.show()

    return {'accuracy': accuracy, 'auc_roc': auc_roc, 'classification_report': report}
