import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(n_estimators=200, learning_rate=0.1, max_depth=5):
    # Load preprocessed data
    X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")
    
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
    accuracy_price = accuracy_score(y_test_price, y_pred_price) * 100
    accuracy_type = accuracy_score(y_test_type, y_pred_type) * 100
    
    report_price = classification_report(y_test_price, y_pred_price)
    report_type = classification_report(y_test_type, y_pred_type)
    
    # Save accuracy & reports to a text file
    with open("static/accuracy.txt", "w") as f:
        f.write(f"Gradient Boosting Accuracy (Price - 5 Classes): {accuracy_price:.2f}%\n")
        f.write(f"Gradient Boosting Accuracy (House Type - 5 Classes): {accuracy_type:.2f}%\n\n")
        f.write("Price Classification Report:\n" + report_price + "\n")
        f.write("House Type Classification Report:\n" + report_type + "\n")
    
    # Confusion Matrices
    cm_price = confusion_matrix(y_test_price, y_pred_price)
    cm_type = confusion_matrix(y_test_type, y_pred_type)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Price Classification")
    plt.savefig("static/confusion_matrix_price.png")
    plt.close()
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - House Type Classification")
    plt.savefig("static/confusion_matrix_type.png")
    plt.close()
    
    # Save Models
    joblib.dump(gb_model_price, "static/gb_price.pkl")
    joblib.dump(gb_model_type, "static/gb_type.pkl")
    
    return accuracy_price, accuracy_type

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_gbc(n_estimators=200, learning_rate=0.1, max_depth=5):
    # Load preprocessed data
    X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

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
    accuracy_price = accuracy_score(y_test_price, y_pred_price) * 100
    accuracy_type = accuracy_score(y_test_type, y_pred_type) * 100

    # Save accuracy to a text file
    with open("static/accuracy.txt", "w") as f:
        f.write(f"Gradient Boosting Accuracy (Price - 5 Classes): {accuracy_price:.2f}%\n")
        f.write(f"Gradient Boosting Accuracy (House Type - 5 Classes): {accuracy_type:.2f}%\n")

    # Confusion Matrix for Price Classification
    cm_price = confusion_matrix(y_test_price, y_pred_price)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Very Low", "Low", "Medium", "High", "Very High"], 
                yticklabels=["Very Low", "Low", "Medium", "High", "Very High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Gradient Boosting - Confusion Matrix (Price - 5 Classes)")
    plt.savefig("static/confusion_matrix_price.png")
    plt.close()

    # Confusion Matrix for House Type Classification
    cm_type = confusion_matrix(y_test_type, y_pred_type)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', 
                xticklabels=["Apartment", "Townhouse", "Villa", "Mansion", "Other"], 
                yticklabels=["Apartment", "Townhouse", "Villa", "Mansion", "Other"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Gradient Boosting - Confusion Matrix (House Type - 5 Classes)")
    plt.savefig("static/confusion_matrix_type.png")
    plt.close()

    # Save Models
    joblib.dump(gb_model_price, "static/gradient_boosting_price_5_classes.pkl")
    joblib.dump(gb_model_type, "static/gradient_boosting_type_5_classes.pkl")

    return accuracy_price, accuracy_type

