import os
import joblib
# For MacOS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV
from dotenv import load_dotenv

# Ensure results directory exists
load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR")
LOG_DIR = os.getenv("LOG_DIR")
LOG_FILE = os.getenv("LOG_FILE")
MODEL_DIR = os.getenv("MODEL_DIR")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_gradient_boosting(n_estimators=200, learning_rate=0.1, max_depth=5,
                            use_grid_search=False, use_random_search=False, use_bayesian_optimization=False):
    """
    Train Gradient Boosting Classifiers for Price and House Type Classification,
    with optional hyperparameter tuning for the Price classifier.
    
    Hyperparameter tuning methods available:
      - Grid Search (use_grid_search=True)
      - Random Search (use_random_search=True)
      - Bayesian Optimization (use_bayesian_optimization=True)
      
    If none of these are enabled, the model uses the default parameters:
      n_estimators, learning_rate, and max_depth.
      
    The House Type classifier is trained with default parameters.
    
    Evaluation metrics (Accuracy, Precision, Recall, F1-score) are computed,
    confusion matrix images are saved, training logs are appended, and a dictionary
    of results (including model paths and metrics) is returned.
    """
    # Load preprocessed data
    preprocessed_data = joblib.load("preprocessed_data.pkl")
    if len(preprocessed_data) == 6:
        X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = preprocessed_data
    else:
        raise ValueError("❌ Preprocessed data format is incorrect. Expected 6 elements but got", len(preprocessed_data))
    
    best_params_price = None

    # --- Hyperparameter Tuning for Price Classification Model ---
    if use_grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train_price)
        gb_model_price = grid_search.best_estimator_
        best_params_price = grid_search.best_params_
    elif use_random_search:
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)
        random_search.fit(X_train, y_train_price)
        gb_model_price = random_search.best_estimator_
        best_params_price = random_search.best_params_
    elif use_bayesian_optimization:
        param_space = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 10)
        }
        base_model = GradientBoostingClassifier(random_state=42)
        bayes_search = BayesSearchCV(base_model, param_space, n_iter=20, cv=3, scoring='accuracy', random_state=42)
        bayes_search.fit(X_train, y_train_price)
        gb_model_price = bayes_search.best_estimator_
        best_params_price = bayes_search.best_params_
    else:
        gb_model_price = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                    max_depth=max_depth, random_state=42)
        gb_model_price.fit(X_train, y_train_price)
        best_params_price = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}

    # --- Train House Type Classification Model with default parameters ---
    gb_model_type = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth, random_state=42)
    gb_model_type.fit(X_train, y_train_type)

    # --- Make Predictions ---
    y_pred_price = gb_model_price.predict(X_test)
    y_pred_type = gb_model_type.predict(X_test)

    # --- Evaluate Metrics for Price Classification ---
    accuracy_price = accuracy_score(y_test_price, y_pred_price)
    precision_price = precision_score(y_test_price, y_pred_price, average='weighted', zero_division=0)
    recall_price = recall_score(y_test_price, y_pred_price, average='weighted', zero_division=0)
    f1_price = f1_score(y_test_price, y_pred_price, average='weighted', zero_division=0)
    classification_report_price = classification_report(y_test_price, y_pred_price, zero_division=0)

    # --- Evaluate Metrics for House Type Classification ---
    accuracy_type = accuracy_score(y_test_type, y_pred_type)
    precision_type = precision_score(y_test_type, y_pred_type, average='weighted', zero_division=0)
    recall_type = recall_score(y_test_type, y_pred_type, average='weighted', zero_division=0)
    f1_type = f1_score(y_test_type, y_pred_type, average='weighted', zero_division=0)
    classification_report_type = classification_report(y_test_type, y_pred_type, zero_division=0)

    print(f"✅ Gradient Boosting Price Model Accuracy: {accuracy_price}")
    print(f"✅ Gradient Boosting House Type Model Accuracy: {accuracy_type}")

    # --- Save Models ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    price_model_path = os.path.join(MODEL_DIR, "gradient_boosting_price.pkl")
    type_model_path = os.path.join(MODEL_DIR, "gradient_boosting_type.pkl")
    joblib.dump(gb_model_price, price_model_path)
    joblib.dump(gb_model_type, type_model_path)

    # --- Generate Confusion Matrices ---
    cm_price = confusion_matrix(y_test_price, y_pred_price)
    cm_type = confusion_matrix(y_test_type, y_pred_type)

    price_cm_file_name = f"confusion_matrix_price_{timestamp}.png"
    type_cm_file_name = f"confusion_matrix_type_{timestamp}.png"
    price_cm_path = os.path.join(RESULTS_DIR, price_cm_file_name)
    type_cm_path = os.path.join(RESULTS_DIR, type_cm_file_name)

    # Plot confusion matrix for Price Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Gradient Boosting - Confusion Matrix (Price - {len(set(y_train_price))} Classes)")
    plt.savefig(price_cm_path)
    plt.close()

    # Plot confusion matrix for House Type Classification
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Gradient Boosting - Confusion Matrix (House Type - {len(set(y_train_type))} Classes)")
    plt.savefig(type_cm_path)
    plt.close()

    print("\n✅ Models and Confusion Matrices saved.")

    # --- Log Training Results ---
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Timestamp: {timestamp};")
        log_file.write("Model: Gradient Boosting;")
        log_file.write("Optimization: ")
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

    # --- Return Results as a Dictionary ---
    return {
        "success": True,
        "accuracy_price": accuracy_price,
        "precision_price": precision_price,
        "recall_price": recall_price,
        "f1_price": f1_price,
        "classification_report_price": classification_report_price,
        "accuracy_type": accuracy_type,
        "precision_type": precision_type,
        "recall_type": recall_type,
        "f1_type": f1_type,
        "classification_report_type": classification_report_type,
        "price_model_path": price_model_path,
        "type_model_path": type_model_path,
        "price_cm_path": price_cm_path,
        "type_cm_path": type_cm_path,
        "training_log": log_file_path,
        "best_params_price": best_params_price
    }

if __name__ == "__main__":
    # Example: Enable one of the tuning methods by setting the corresponding flag to True.
    results = train_gradient_boosting(n_estimators=200, learning_rate=0.1, max_depth=5,
                                      use_grid_search=False, use_random_search=True, use_bayesian_optimization=False)
    print("Training results:", results)