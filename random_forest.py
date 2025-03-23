import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure results directory exists
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_random_forest(n_estimators=200, max_depth=10, min_samples_split=2):
    """
    Train Random Forest Classifier for Price and House Type Classification,
    save confusion matrix images, and return evaluation results.
    """

    # Load preprocessed data
    preprocessed_data = joblib.load("preprocessed_data.pkl")

    if len(preprocessed_data) == 6:
        X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = preprocessed_data
    else:
        raise ValueError("❌ Preprocessed data format is incorrect. Expected 6 elements but got", len(preprocessed_data))

    # Train Random Forest for Price Classification (Dynamic Classes)
    rf_model_price = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf_model_price.fit(X_train, y_train_price)

    # Train Random Forest for House Type Classification (Dynamic Classes)
    rf_model_type = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf_model_type.fit(X_train, y_train_type)

    # Make Predictions
    y_pred_price = rf_model_price.predict(X_test)
    y_pred_type = rf_model_type.predict(X_test)

    # Evaluate Models
    accuracy_price = accuracy_score(y_test_price, y_pred_price)
    accuracy_type = accuracy_score(y_test_type, y_pred_type)

    classification_report_price = classification_report(y_test_price, y_pred_price)
    classification_report_type = classification_report(y_test_type, y_pred_type)

    print(f"✅ Random Forest Accuracy (Price): {accuracy_price}")
    print(f"✅ Random Forest Accuracy (House Type): {accuracy_type}")

    # Save models
    price_model_path = os.path.join(RESULTS_DIR, "random_forest_price.pkl")
    type_model_path = os.path.join(RESULTS_DIR, "random_forest_type.pkl")
    joblib.dump(rf_model_price, price_model_path)
    joblib.dump(rf_model_type, type_model_path)

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
    plt.title(f"Random Forest - Confusion Matrix (Price - {len(set(y_train_price))} Classes)")
    plt.savefig(price_cm_path)  # Save image
    plt.close()

    # Confusion Matrix for House Type Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Random Forest - Confusion Matrix (House Type - {len(set(y_train_type))} Classes)")
    plt.savefig(type_cm_path)  # Save image
    plt.close()

    print("\n✅ Models and Confusion Matrices saved.")

    # Save training results in a text file
    log_file_path = os.path.join(RESULTS_DIR, f"training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Training Timestamp: {timestamp}\n")
        log_file.write(f"Random Forest Accuracy (House Price): {accuracy_price:.4f}\n")
        log_file.write(f"Random Forest Accuracy (House Type): {accuracy_type:.4f}\n\n")
        log_file.write("Price Classification Report:\n")
        log_file.write(classification_report_price + "\n\n")
        log_file.write("House Type Classification Report:\n")
        log_file.write(classification_report_type + "\n")

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
