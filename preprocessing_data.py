# preprocessing_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving processed data

# Load dataset
df = pd.read_csv("Housing_Augmented.csv")  # Replace with actual file path

# Convert Binary Categorical Variables (yes/no → 1/0)
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == "yes" else 0)

# One-Hot Encoding for 'furnishingstatus'
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# Define columns for which we want to categorize price
target_columns = ["area", "bedrooms", "bathrooms", "stories"]

# Function to categorize price based on percentiles per column
def categorize_price_by_feature(df, column):
    """Assigns price categories based on percentiles for a given feature."""
    if column not in df.columns or column == "price":
        return None

    # Compute percentiles for each unique value in the column
    price_categories = {}
    for value in df[column].unique():
        subset = df[df[column] == value]['price']
        if len(subset) < 5:  # Skip small groups
            continue
        
        q1, q2, q3, q4 = subset.quantile([0.2, 0.4, 0.6, 0.8])
        price_categories[value] = (q1, q2, q3, q4)

    # Assign categories based on thresholds
    def assign_category(row):
        if row[column] in price_categories:
            q1, q2, q3, q4 = price_categories[row[column]]
            if row['price'] <= q1:
                return 0
            elif row['price'] <= q2:
                return 1
            elif row['price'] <= q3:
                return 2
            elif row['price'] <= q4:
                return 3
            else:
                return 4
        return 2  # Default to middle category if no data

    return df.apply(assign_category, axis=1)

# Apply price categorization for selected columns
price_category_cols = {}
for column in target_columns:
    price_category_cols[f'price_category_by_{column}'] = categorize_price_by_feature(df, column)

# Add new price category columns to DataFrame
df = df.assign(**price_category_cols)

# Compute General Price Category Based on Overall Price Percentiles
q1 = df['price'].quantile(0.20)  # 20th percentile
q2 = df['price'].quantile(0.40)  # 40th percentile
q3 = df['price'].quantile(0.60)  # 60th percentile
q4 = df['price'].quantile(0.80)  # 80th percentile

df['price_category'] = df['price'].apply(
    lambda x: 0 if x <= q1 else (1 if x <= q2 else (2 if x <= q3 else (3 if x <= q4 else 4)))
)

# Classify House Type into 5 Categories
def house_type(row):
    if row['area'] <= 2000 and row['bedrooms'] == 1 and row['bathrooms'] <= 1:
        return 0  # Studio
    elif 2000 < row['area'] <= 4000 and row['bedrooms'] in [1, 2] and row['stories'] <= 2:
        return 1  # Apartment
    elif 4000 < row['area'] <= 8000 and row['bedrooms'] in [2, 3, 4] and row['stories'] == 2:
        return 2  # Townhouse
    elif 8000 < row['area'] <= 12000 and row['bedrooms'] in [4, 5] and row['stories'] >= 2:
        return 3  # Villa
    else:
        return 4  # Mansion

df['house_type'] = df.apply(house_type, axis=1)

# Prepare Features & Target Variables
X = df.drop(columns=["price", "price_category", "house_type"])  # Features
y_price = df["price_category"]  # Target for Price Classification (5 categories)
y_type = df["house_type"]  # Target for House Type Classification (5 categories)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data into Training & Testing Sets
X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = train_test_split(
    X_scaled, y_price, y_type, test_size=0.2, random_state=42, stratify=y_price
)

# Save Preprocessed Data
joblib.dump((X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type), "preprocessed_data.pkl")
print("✅ Preprocessed data saved as 'preprocessed_data.pkl'")
