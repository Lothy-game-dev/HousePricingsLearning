# preprocessing_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import joblib  # For saving processed data

# Load dataset
df = pd.read_csv("Housing.csv")  # Replace with actual file path

# Convert Binary Categorical Variables (yes/no → 1/0)
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == "yes" else 0)

# One-Hot Encoding for 'furnishingstatus'
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# Convert 'price' into 5 categories using KBinsDiscretizer (Equal-sized bins)
price_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['price_category'] = price_discretizer.fit_transform(df[['price']]).astype(int)

# Classify House Type based on Area & Bedrooms
def house_type(row):
    if row['area'] <= 2500 and row['bedrooms'] <= 2:
        return 0  # Apartment
    elif 2500 < row['area'] <= 6000 and row['bedrooms'] in [3, 4]:
        return 1  # Townhouse
    else:
        return 2  # Villa

df['house_type'] = df.apply(house_type, axis=1)

# Prepare Features & Target Variables
X = df.drop(columns=["price", "price_category", "house_type"])  # Features
y_price = df["price_category"]  # Target for Price Classification (now 5 classes)
y_type = df["house_type"]  # Target for House Type Classification

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = train_test_split(
    X_scaled, y_price, y_type, test_size=0.2, random_state=42, stratify=y_price
)

# Save Preprocessed Data
joblib.dump((X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type), "preprocessed_data.pkl")
print("Preprocessed data saved as 'preprocessed_data.pkl'")
