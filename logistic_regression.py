import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure results directory exists
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_logistic_regression(max_iter=500, solver='lbfgs'):
    """
    Train Logistic Regression for Price and House Type Classification,
    save confusion matrix images, and return evaluation results.
    """

    # Load preprocessed data
    preprocessed_data = joblib.load("preprocessed_data.pkl")

    if len(preprocessed_data) == 6:
        X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = preprocessed_data
    else:
        raise ValueError("❌ Preprocessed data format is incorrect. Expected 6 elements but got", len(preprocessed_data))

    # Train Logistic Regression for Price Classification
    log_model_price = LogisticRegression(max_iter=max_iter, multi_class='multinomial', solver=solver)
    log_model_price.fit(X_train, y_train_price)

    # Train Logistic Regression for House Type Classification
    log_model_type = LogisticRegression(max_iter=max_iter, multi_class='ovr', solver=solver)
    log_model_type.fit(X_train, y_train_type)

    # Make Predictions
    y_pred_price = log_model_price.predict(X_test)
    y_pred_type = log_model_type.predict(X_test)

    # Evaluate Models
    accuracy_price = accuracy_score(y_test_price, y_pred_price)
    accuracy_type = accuracy_score(y_test_type, y_pred_type)

    classification_report_price = classification_report(y_test_price, y_pred_price)
    classification_report_type = classification_report(y_test_type, y_pred_type)

    print(f"✅ Logistic Regression Accuracy (Price - {len(set(y_train_price))} Classes): {accuracy_price}")
    print(f"✅ Logistic Regression Accuracy (House Type - {len(set(y_train_type))} Classes): {accuracy_type}")

    # Save models
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    price_model_path = os.path.join(RESULTS_DIR, f"logistic_regression_price_{timestamp}.pkl")
    type_model_path = os.path.join(RESULTS_DIR, f"logistic_regression_type_{timestamp}.pkl")
    joblib.dump(log_model_price, price_model_path)
    joblib.dump(log_model_type, type_model_path)

    # Generate Confusion Matrices
    cm_price = confusion_matrix(y_test_price, y_pred_price)
    cm_type = confusion_matrix(y_test_type, y_pred_type)

    price_cm_file_name = f"confusion_matrix_price_{timestamp}.png"
    type_cm_file_name = f"confusion_matrix_type_{timestamp}.png"

    price_cm_path = os.path.join(RESULTS_DIR, price_cm_file_name)
    type_cm_path = os.path.join(RESULTS_DIR, type_cm_file_name)

    # Confusion Matrix for Price Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Logistic Regression - Confusion Matrix (Price - {len(set(y_train_price))} Classes)")
    plt.savefig(price_cm_path)  # Save image
    plt.close()

    # Confusion Matrix for House Type Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Logistic Regression - Confusion Matrix (House Type - {len(set(y_train_type))} Classes)")
    plt.savefig(type_cm_path)  # Save image
    plt.close()

    print("\n✅ Models and Confusion Matrices saved.")

    # Save training results in a text file
    log_file_path = os.path.join(RESULTS_DIR, "training_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Timestamp: {timestamp};")
        log_file.write(f"Model: Logistic Regression;")
        log_file.write(f"Hyperparameters: max_iter={max_iter},solver={solver};")
        log_file.write(f"Price: {accuracy_price:.4f};")
        log_file.write(f"Type: {accuracy_type:.4f}\n")

    # Return results as a dictionary
    return {
        "success": True,
        "accuracy_price": accuracy_price,
        "accuracy_type": accuracy_type,
        "classification_report_price": classification_report_price,
        "classification_report_type": classification_report_type,
        "price_model_path": price_model_path,
        "type_model_path": type_model_path,
        "price_cm_path": price_cm_path,
        "type_cm_path": type_cm_path,
        "training_log": log_file_path
    }
