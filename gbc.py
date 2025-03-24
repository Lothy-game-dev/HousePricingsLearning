import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure results directory exists
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_gradient_boosting(n_estimators=200, learning_rate=0.1, max_depth=5):
    """
    Train Gradient Boosting Classifier for Price and House Type Classification,
    save confusion matrix images, and return evaluation results.
    """

    # Load preprocessed data
    preprocessed_data = joblib.load("preprocessed_data.pkl")

    if len(preprocessed_data) == 6:
        X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = preprocessed_data
    else:
        raise ValueError("❌ Preprocessed data format is incorrect. Expected 6 elements but got", len(preprocessed_data))

    # Train Gradient Boosting for Price Classification
    gb_model_price = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    gb_model_price.fit(X_train, y_train_price)

    # Train Gradient Boosting for House Type Classification
    gb_model_type = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    gb_model_type.fit(X_train, y_train_type)

    # Make Predictions
    y_pred_price = gb_model_price.predict(X_test)
    y_pred_type = gb_model_type.predict(X_test)

    # Evaluate Models
    accuracy_price = accuracy_score(y_test_price, y_pred_price)
    accuracy_type = accuracy_score(y_test_type, y_pred_type)

    classification_report_price = classification_report(y_test_price, y_pred_price)
    classification_report_type = classification_report(y_test_type, y_pred_type)

    print(f"✅ Gradient Boosting Accuracy (Price): {accuracy_price}")
    print(f"✅ Gradient Boosting Accuracy (House Type): {accuracy_type}")

    # Save models
    price_model_path = os.path.join(RESULTS_DIR, "gradient_boosting_price.pkl")
    type_model_path = os.path.join(RESULTS_DIR, "gradient_boosting_type.pkl")
    joblib.dump(gb_model_price, price_model_path)
    joblib.dump(gb_model_type, type_model_path)

    # Generate Confusion Matrices
    cm_price = confusion_matrix(y_test_price, y_pred_price)
    cm_type = confusion_matrix(y_test_type, y_pred_type)

    # Generate unique timestamp for this training session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    price_cm_file_name = f"confusion_matrix_price_{timestamp}.png"
    type_cm_file_name = f"confusion_matrix_type_{timestamp}.png"

    price_cm_path = os.path.join(RESULTS_DIR, price_cm_file_name)
    type_cm_path = os.path.join(RESULTS_DIR, type_cm_file_name)

    # Confusion Matrix for Price Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Gradient Boosting - Confusion Matrix (Price - {len(set(y_train_price))} Classes)")
    plt.savefig(price_cm_path)  # Save image
    plt.close()

    # Confusion Matrix for House Type Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Gradient Boosting - Confusion Matrix (House Type - {len(set(y_train_type))} Classes)")
    plt.savefig(type_cm_path)  # Save image
    plt.close()

    print("\n✅ Models and Confusion Matrices saved.")

    # Save training results in a text file
    log_file_path = os.path.join(RESULTS_DIR, f"training_log.txt")
    with open(log_file_path, "a") as log_file:  # Append new training logs
        log_file.write(f"Timestamp: {timestamp};")
        log_file.write(f"Model: Gradient Boosting;")
        log_file.write(f"Hyperparameters: n_estimators={n_estimators},learning_rate={learning_rate},max_depth={max_depth};")
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
