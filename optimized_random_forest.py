# optimized_random_forest.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

# Feature Selection: Keep Only the 10 Best Features
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(rf_temp, n_features_to_select=10)
X_train_selected = selector.fit_transform(X_train, y_train_price)
X_test_selected = selector.transform(X_test)

# Define Hyperparameter Search Space
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples to split
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'bootstrap': [True, False],  # Use bootstrapping
    'class_weight': ['balanced', None]  # Handle class imbalance
}

# Perform Randomized Search for Price Classification
random_search_price = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=10,  # Limits number of combinations tested
    cv=5,  # 5-Fold Cross Validation
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search_price.fit(X_train_selected, y_train_price)
best_rf_model_price = random_search_price.best_estimator_

# Perform Randomized Search for House Type Classification
random_search_type = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=10,  # Limits number of combinations tested
    cv=5,  # 5-Fold Cross Validation
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search_type.fit(X_train_selected, y_train_type)
best_rf_model_type = random_search_type.best_estimator_

# Make Predictions
y_pred_price = best_rf_model_price.predict(X_test_selected)
y_pred_type = best_rf_model_type.predict(X_test_selected)

# Evaluate Models
accuracy_price = accuracy_score(y_test_price, y_pred_price)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

print(f"Optimized Random Forest Accuracy (Price - 3 Classes): {accuracy_price}")
print(f"Optimized Random Forest Accuracy (House Type - 3 Classes): {accuracy_type}")

print("\nBest Parameters for Price Classification:\n", random_search_price.best_params_)
print("\nBest Parameters for House Type Classification:\n", random_search_type.best_params_)

print("\nPrice Classification Report (3 Classes):\n", classification_report(y_test_price, y_pred_price))
print("\nHouse Type Classification Report (3 Classes):\n", classification_report(y_test_type, y_pred_type))

# Confusion Matrix for Price Classification (3 Classes)
cm_price = confusion_matrix(y_test_price, y_pred_price)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized Random Forest - Confusion Matrix (Price - 3 Classes)")
plt.show()

# Confusion Matrix for House Type Classification (3 Classes)
cm_type = confusion_matrix(y_test_type, y_pred_type)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', xticklabels=["Apartment", "Townhouse", "Villa"], yticklabels=["Apartment", "Townhouse", "Villa"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized Random Forest - Confusion Matrix (House Type - 3 Classes)")
plt.show()

# Save Optimized Models
joblib.dump(best_rf_model_price, "optimized_rf_price.pkl")
joblib.dump(best_rf_model_type, "optimized_rf_type.pkl")

print("\n✅ Optimized models saved as 'optimized_rf_price.pkl' and 'optimized_rf_type.pkl'")
