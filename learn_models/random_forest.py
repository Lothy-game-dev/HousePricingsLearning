import os
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV
from dotenv import load_dotenv

load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR")
LOG_DIR = os.getenv("LOG_DIR")
LOG_FILE = os.getenv("LOG_FILE")
MODEL_DIR = os.getenv("MODEL_DIR")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_random_forest(n_estimators=200, max_depth=10, min_samples_split=2,
                        use_grid_search=False, use_random_search=False, use_bayesian_optimization=False):
    """
    Train a Random Forest Classifier for Price and House Type Classification.
    Optionally, perform hyperparameter tuning using one of:
      - Grid Search (use_grid_search=True)
      - Random Search (use_random_search=True)
      - Bayesian Optimization (use_bayesian_optimization=True)
    Saves the trained models, confusion matrix images, and logs evaluation metrics including
    Accuracy, Precision, Recall, and F1-score. Returns a dictionary of results.
    """
    # Load preprocessed data
    preprocessed_data = joblib.load("preprocessed_data.pkl")
    if len(preprocessed_data) == 6:
        X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = preprocessed_data
    else:
        raise ValueError("❌ Preprocessed data format is incorrect. Expected 6 elements but got", len(preprocessed_data))
    
    best_params_price = None

    # ----- Hyperparameter Tuning for Price Classification Model -----
    if use_grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [max_depth, 10, 20],
            'min_samples_split': [min_samples_split, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train_price)
        rf_model_price = grid_search.best_estimator_
        best_params_price = grid_search.best_params_
    elif use_random_search:
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, min_samples_split, 4, 6]
        }
        base_model = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)
        random_search.fit(X_train, y_train_price)
        rf_model_price = random_search.best_estimator_
        best_params_price = random_search.best_params_
    elif use_bayesian_optimization:
        param_space = {
            'n_estimators': (100, 500),
            'max_depth': (5, 30),
            'min_samples_split': (2, 10)
        }
        base_model = RandomForestClassifier(random_state=42)
        bayes_search = BayesSearchCV(base_model, param_space, n_iter=20, cv=3, scoring='accuracy', random_state=42)
        bayes_search.fit(X_train, y_train_price)
        rf_model_price = bayes_search.best_estimator_
        best_params_price = bayes_search.best_params_
    else:
        # Default training without hyperparameter tuning
        rf_model_price = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  min_samples_split=min_samples_split, random_state=42)
        rf_model_price.fit(X_train, y_train_price)
        best_params_price = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}

    # ----- Train the House Type Classification Model (default parameters) -----
    rf_model_type = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, random_state=42)
    rf_model_type.fit(X_train, y_train_type)

    # Make Predictions
    y_pred_price = rf_model_price.predict(X_test)
    y_pred_type = rf_model_type.predict(X_test)

    # Evaluate Price Classification
    accuracy_price = accuracy_score(y_test_price, y_pred_price)
    precision_price = precision_score(y_test_price, y_pred_price, average='weighted', zero_division=0)
    recall_price = recall_score(y_test_price, y_pred_price, average='weighted', zero_division=0)
    f1_price = f1_score(y_test_price, y_pred_price, average='weighted', zero_division=0)

    # Evaluate House Type Classification
    accuracy_type = accuracy_score(y_test_type, y_pred_type)
    precision_type = precision_score(y_test_type, y_pred_type, average='weighted', zero_division=0)
    recall_type = recall_score(y_test_type, y_pred_type, average='weighted', zero_division=0)
    f1_type = f1_score(y_test_type, y_pred_type, average='weighted', zero_division=0)

    classification_report_price = classification_report(y_test_price, y_pred_price, zero_division=0)
    classification_report_type = classification_report(y_test_type, y_pred_type, zero_division=0)

    print(f"✅ Gradient Boosting Accuracy (Price): {accuracy_price}")
    print(f"✅ Gradient Boosting Accuracy (House Type): {accuracy_type}")

    # Save models
    price_model_path = os.path.join(MODEL_DIR, "random_forest_price.pkl")
    type_model_path = os.path.join(MODEL_DIR, "random_forest_type.pkl")
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
    plt.savefig(price_cm_path)
    plt.close()

    # Confusion Matrix for House Type Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Random Forest - Confusion Matrix (House Type - {len(set(y_train_type))} Classes)")
    plt.savefig(type_cm_path)
    plt.close()

    print("\n✅ Models and Confusion Matrices saved.")

    # Save training results in a text file
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Timestamp: {timestamp};")
        log_file.write("Model: Random Forest;")
        log_file.write(f"Optimization: ")
        if use_grid_search:
            log_file.write("Grid Search;")
        elif use_random_search:
            log_file.write("Random Search;")
        elif use_bayesian_optimization:
            log_file.write("Bayesian Optimization;")
        else:
            log_file.write("None;")
        log_file.write(f"Hyperparameters: {best_params_price};")
        log_file.write(f"Price: Accuracy={accuracy_price:.4f}, Precision={precision_price:.4f}, Recall={recall_price:.4f}, F1={f1_price:.4f};")
        log_file.write(f"Type: Accuracy={accuracy_type:.4f}, Precision={precision_type:.4f}, Recall={recall_type:.4f}, F1={f1_type:.4f}\n")

    # Return results as a dictionary
    return {
        "success": True,
        "accuracy_price": accuracy_price,
        "precision_price": precision_price,
        "recall_price": recall_price,
        "f1_price": f1_price,
        "accuracy_type": accuracy_type,
        "precision_type": precision_type,
        "recall_type": recall_type,
        "f1_type": f1_type,
        "classification_report_price": classification_report_price,
        "classification_report_type": classification_report_type,
        "price_model_path": price_model_path,
        "type_model_path": type_model_path,
        "price_cm_path": price_cm_path,
        "type_cm_path": type_cm_path,
        "training_log": log_file_path,
        "best_params_price": best_params_price
    }

if __name__ == "__main__":
    results = train_random_forest(use_grid_search=False, use_random_search=True, use_bayesian_optimization=False)
    print("Training results:", results)