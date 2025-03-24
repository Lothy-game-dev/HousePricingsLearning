import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

def preprocess_data(num_price_classes=5, num_house_types=5):
    """Preprocess dataset dynamically, ensuring numerical data is correctly formatted and stratification works."""

    # Load dataset
    df = pd.read_csv("Housing_Augmented.csv")

    # Load environment variables
    load_dotenv()

    # Read contribution values from `.env`
    FEATURE_WEIGHTS = {
        "area": float(os.getenv("AREA_CONTRIBUTION", 17)) / 1000,
        "prefarea": int(os.getenv("PREFAREA_CONTRIBUTION", 10)),
        "bedrooms": int(os.getenv("BEDROOMS_CONTRIBUTION", 5)),
        "bathrooms": int(os.getenv("BATHROOMS_CONTRIBUTION", 5)),
        "parking": int(os.getenv("PARKING_CONTRIBUTION", 3)),
        "stories": int(os.getenv("STORIES_CONTRIBUTION", 3)),
        "airconditioning": int(os.getenv("AIRCONDITIONING_CONTRIBUTION", 3)),
        "basement": int(os.getenv("BASEMENT_CONTRIBUTION", 2)),
        "guestroom": int(os.getenv("GUESTROOM_CONTRIBUTION", 2)),
        "mainroad": int(os.getenv("MAINROAD_CONTRIBUTION", -2)),
        "hotwaterheating": int(os.getenv("HOTWATERHEATING_CONTRIBUTION", 1)),
    }

    # Convert Binary Categorical Variables (yes/no → 1/0)
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # One-Hot Encoding for 'furnishingstatus'
    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    # Ensure required columns exist (avoid KeyErrors)
    df['furnishingstatus_furnished'] = df.get('furnishingstatus_furnished', 0)
    df['furnishingstatus_semi-furnished'] = df.get('furnishingstatus_semi-furnished', 0)

    # Convert all numerical columns to int/float
    numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, replace errors with NaN

    # Drop rows with NaN values caused by conversion issues
    df = df.dropna()

    # **House Type Classification Using Weighted Features from `.env`**
    df["house_score"] = sum(df[col].astype(float) * FEATURE_WEIGHTS[col] for col in FEATURE_WEIGHTS if col in df.columns)

    # Ensure Exactly 5 Classes for House Type
    house_bins = np.linspace(df["house_score"].min(), df["house_score"].max(), num=num_house_types+1)
    df["house_type"] = np.digitize(df["house_score"], bins=house_bins[:-1], right=True) - 1  # Ensure class range [0,4]

    target_columns = [col for col in FEATURE_WEIGHTS if col in df.columns]

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

    # **Prepare Features & Target Variables**
    X = df.drop(columns=["price", "price_category", "house_type", "house_score"])
    y_price = df["price_category"]
    y_type = df["house_type"]

    # **Standardize Features**
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type = train_test_split(
        X_scaled, y_price, y_type, test_size=0.2, random_state=42, stratify=y_price
    )

    # **Save Preprocessed Data**
    joblib.dump((X_train, X_test, y_train_price, y_test_price, y_train_type, y_test_type), "preprocessed_data.pkl")

    # **Return Sample Data for UI Display**
    df_sample = pd.DataFrame(X_train[:10])
    df_sample["price_category"] = y_train_price[:10].tolist()
    df_sample["house_type"] = y_train_type[:10].tolist()
    
    return df_sample.to_dict(orient="records")
