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
    df = pd.read_csv("Housing.csv")

    # Load environment variables
    load_dotenv()

    # Read contribution values from `.env`
    FEATURE_WEIGHTS = {
        "area": float(os.getenv("AREA_CONTRIBUTION", 17)) / 1000,
        "prefarea": int(os.getenv("PREFAREA_CONTRIBUTION", 10)),
        "bedrooms": int(os.getenv("BEDROOMS_CONTRIBUTION", 5)),
        "bathrooms": int(os.getenv("BATHROOMS_CONTRIBUTION", 5)),
        "parking": int(os.getenv("PARKING_CONTRIBUTION", 3)),
        "furnishingstatus_furnished": int(os.getenv("FURNISHING_FURNISHED_CONTRIBUTION", 3)),
        "furnishingstatus_semi-furnished": int(os.getenv("FURNISHING_SEMI_FURNISHED_CONTRIBUTION", 3)),
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

    def categorize_price_by_house_type(df, num_price_classes=5):
        """Categorizes price based on percentiles within each house type."""
        price_bins_per_type = {}

        # Compute percentiles within each house type
        for house_type in df["house_type"].unique():
            subset = df[df["house_type"] == house_type]["price"]
            if len(subset) < num_price_classes:  # Avoid too few values in a category
                continue
            
            price_bins = subset.quantile(np.linspace(0, 1, num_price_classes+1)).values
            price_bins_per_type[house_type] = price_bins

        # Assign categories based on house type bins
        def assign_price_category(row):
            house_type = row["house_type"]
            if house_type in price_bins_per_type:
                bins = price_bins_per_type[house_type]
                return np.digitize(row["price"], bins[:-1], right=True) - 1  # Ensure class range [0,4]
            return np.nan  # Default to NaN if house type is missing

        return df.apply(assign_price_category, axis=1), price_bins_per_type

    # Apply the price categorization method
    df["price_category"], price_bins_per_house_type = categorize_price_by_house_type(df, num_price_classes)

    # **Fix NaN Values in `price_category` (Avoids `NaN` in Y-Values)**
    df["price_category"].fillna(df["price_category"].median(), inplace=True)
    df = df.dropna(subset=["price_category"])  # Drop rows where price category is still NaN

    # **Ensure Exactly 5 Price Classes**
    df["price_category"] = df["price_category"].clip(0, num_price_classes - 1)  # Ensures class range [0,4]

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
    
    return df_sample.to_dict(orient="records"), list(map(str, price_bins_per_house_type))
