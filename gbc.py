import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = joblib.load("preprocessed_data.pkl")

# Train Gradient Boosting for Price Classification (5 classes)
gbm_price = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42)
gbm_price.fit(X_train, y_train_price)

# Train Gradient Boosting for House Type Classification (3 classes)
gbm_type = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
gbm_type.fit(X_train, y_train_type)

# Make Predictions
y_pred_price = gbm_price.predict(X_test)
y_pred_type = gbm_type.predict(X_test)

# Evaluate Models
accuracy_price = accuracy_score(y_test_price, y_pred_price)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

print(f"Gradient Boosting Accuracy (Price - 3 Classes): {accuracy_price}")
print(f"Gradient Boosting Accuracy (House Type): {accuracy_type}")

print("\nPrice Classification Report (3 Classes):\n", classification_report(y_test_price, y_pred_price))
print("\nHouse Type Classification Report:\n", classification_report(y_test_type, y_pred_type))

# Confusion Matrix for Price Classification (3 classes)
cm_price = confusion_matrix(y_test_price, y_pred_price)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting - Confusion Matrix (Price - 3 Classes)")
plt.show()

# Confusion Matrix for House Type Classification
cm_type = confusion_matrix(y_test_type, y_pred_type)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_type, annot=True, fmt='d', cmap='Greens', xticklabels=["Apartment", "Townhouse", "Villa"], yticklabels=["Apartment", "Townhouse", "Villa"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting - Confusion Matrix (House Type)")
plt.show()

# Save Models
joblib.dump(gbm_price, "gradient_boosting_price.pkl")
joblib.dump(gbm_type, "gradient_boosting_type.pkl")

print("\nModels saved as gradient_boosting_price.pkl and gradient_boosting_type.pkl")
