import pandas as pd
import random
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CSV_GENERATION_DIR = os.getenv("CSV_GENERATION_DIR")

os.makedirs(CSV_GENERATION_DIR, exist_ok=True)

def generate_random_data_for_prediction(file_path="Housing_Augmented.csv", n=10):
    df = pd.read_csv(file_path)

    # Columns to ignore
    ignore_cols = ["price", "price_category", "house_type", "house_score"]

    # Convert yes/no columns to lowercase for consistency
    binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in binary_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Prepare usable features
    usable_cols = [col for col in df.columns if col not in ignore_cols]

    # Get ranges for numeric columns
    numeric_cols = df[usable_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_ranges = {
        col: (int(df[col].min()), int(df[col].max()))
        for col in numeric_cols
    }

    # Categorical / binary values
    categorical_values = {
        "furnishingstatus": df["furnishingstatus"].dropna().unique().tolist()
    }
    binary_values = {col: ["yes", "no"] for col in binary_columns}

    # Generate data
    records = []
    for _ in range(n):
        row = {}
        for col in numeric_cols:
            row[col] = random.randint(*numeric_ranges[col])
        for col in binary_columns:
            row[col] = random.choice(binary_values[col])
        row["furnishingstatus"] = random.choice(categorical_values["furnishingstatus"])
        records.append(row)

    filePath = os.path.join(CSV_GENERATION_DIR, f"random_input_for_prediction_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    returnData = pd.DataFrame(records)
    returnData.to_csv(filePath, index=False)

    return filePath, returnData.to_dict()