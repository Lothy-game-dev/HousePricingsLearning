# logistic_regression.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

# Train Logistic Regression for Price Classification (Now 5 classes)
log_model_price = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs')
log_model_price.fit(X_train, y_train_price)

# Train Logistic Regression for House Type Classification
log_model_type = LogisticRegression(max_iter=500, multi_class='ovr', solver='lbfgs')
log_model_type.fit(X_train, y_train_type)

# Make Predictions
y_pred_price = log_model_price.predict(X_test)
y_pred_type = log_model_type.predict(X_test)

# Evaluate Models
accuracy_price = accuracy_score(y_test_price, y_pred_price)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

print(f"Logistic Regression Accuracy (Price - 5 Classes): {accuracy_price}")
print(f"Logistic Regression Accuracy (House Type): {accuracy_type}")

print("\nPrice Classification Report (5 Classes):\n", classification_report(y_test_price, y_pred_price))
print("\nHouse Type Classification Report:\n", classification_report(y_test_type, y_pred_type))

# Confusion Matrix for Price Classification (5 classes)
cm_price = confusion_matrix(y_test_price, y_pred_price)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix (Price - 5 Classes)")
plt.show()

# Confusion Matrix for House Type Classification
cm_type = confusion_matrix(y_test_type, y_pred_type)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', xticklabels=["Apartment", "Townhouse", "Villa"], yticklabels=["Apartment", "Townhouse", "Villa"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix (House Type)")
plt.show()

# Save Models
joblib.dump(log_model_price, "logistic_regression_price.pkl")
joblib.dump(log_model_type, "logistic_regression_type.pkl")

print("\nModels saved as logistic_regression_price.pkl and logistic_regression_type.pkl")
