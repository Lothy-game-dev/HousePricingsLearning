# random_forest.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

# Train Random Forest for Price Classification (Now 5 classes)
rf_model_price = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model_price.fit(X_train, y_train_price)

# Train Random Forest for House Type Classification (5 classes)
rf_model_type = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model_type.fit(X_train, y_train_type)

# Make Predictions
y_pred_price = rf_model_price.predict(X_test)
y_pred_type = rf_model_type.predict(X_test)

# Evaluate Models
accuracy_price = accuracy_score(y_test_price, y_pred_price)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

print(f"Random Forest Accuracy (Price - 5 Classes): {accuracy_price}")
print(f"Random Forest Accuracy (House Type - 5 Classes): {accuracy_type}")

print("\nPrice Classification Report (5 Classes):\n", classification_report(y_test_price, y_pred_price))
print("\nHouse Type Classification Report (5 Classes):\n", classification_report(y_test_type, y_pred_type))

# Confusion Matrix for Price Classification (5 Classes)
cm_price = confusion_matrix(y_test_price, y_pred_price)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues', xticklabels=["Very Low", "Low", "Medium", "High", "Very High"], yticklabels=["Very Low", "Low", "Medium", "High", "Very High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix (Price - 5 Classes)")
plt.show()

# Confusion Matrix for House Type Classification (5 Classes)
cm_type = confusion_matrix(y_test_type, y_pred_type)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', xticklabels=["Studio", "Apartment", "Townhouse", "Villa", "Mansion"], yticklabels=["Studio", "Apartment", "Townhouse", "Villa", "Mansion"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix (House Type - 5 Classes)")
plt.show()

# Save Models
joblib.dump(rf_model_price, "random_forest_price_5_classes.pkl")
joblib.dump(rf_model_type, "random_forest_type_5_classes.pkl")

print("\n✅ Models saved as 'random_forest_price_5_classes.pkl' and 'random_forest_type_5_classes.pkl'")
