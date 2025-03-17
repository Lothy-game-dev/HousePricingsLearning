# random_forest.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

# Train Random Forest for Price Classification (Now 10 classes)
rf_model_price = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model_price.fit(X_train, y_train_price)

# Train Random Forest for House Type Classification
rf_model_type = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model_type.fit(X_train, y_train_type)

# Make Predictions
y_pred_price = rf_model_price.predict(X_test)
y_pred_type = rf_model_type.predict(X_test)

# Evaluate Models
accuracy_price = accuracy_score(y_test_price, y_pred_price)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

print(f"Random Forest Accuracy (Price - 10 classes): {accuracy_price}")
print(f"Random Forest Accuracy (House Type): {accuracy_type}")

print("\nPrice Classification Report (10 Classes):\n", classification_report(y_test_price, y_pred_price))
print("\nHouse Type Classification Report:\n", classification_report(y_test_type, y_pred_type))

# Confusion Matrix for Price Classification (10 classes)
cm_price = confusion_matrix(y_test_price, y_pred_price)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix (Price - 10 Classes)")
plt.show()

# Confusion Matrix for House Type Classification
cm_type = confusion_matrix(y_test_type, y_pred_type)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', xticklabels=["Apartment", "Townhouse", "Villa"], yticklabels=["Apartment", "Townhouse", "Villa"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix (House Type)")
plt.show()

# Save Models
joblib.dump(rf_model_price, "random_forest_price.pkl")
joblib.dump(rf_model_type, "random_forest_type.pkl")

print("\nModels saved as random_forest_price.pkl and random_forest_type.pkl")
