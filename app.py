from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import joblib
import threading
import preprocessing_data as preprocessing
import random_forest
import models
import pandas as pd
from dotenv import load_dotenv, set_key, dotenv_values
import os

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("Housing_Augmented.csv")

# Define available models and their hyperparameters
AVAILABLE_MODELS = {
    "logistic_regression": {
        "C": {"type": "float", "default": 1.0, "description": "Inverse of regularization strength"},
        "max_iter": {"type": "int", "default": 100, "description": "Maximum number of iterations"},
        "solver": {"type": "str", "default": "lbfgs", "options": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
    },
    "random_forest": {
        "n_estimators": {"type": "int", "default": 200, "description": "Number of trees in the forest"},
        "max_depth": {"type": "int", "default": 10, "description": "Maximum depth of the tree"},
        "min_samples_split": {"type": "int", "default": 2, "description": "Minimum samples required to split a node"}
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "default": 100, "description": "Number of boosting stages"},
        "learning_rate": {"type": "float", "default": 0.1, "description": "Step size shrinkage for boosting"},
        "max_depth": {"type": "int", "default": 3, "description": "Maximum depth of the tree"}
    }
}

@app.route("/get-available-models", methods=["GET"])
def get_available_models():
    """Return available models and their hyperparameters"""
    return jsonify({"models": AVAILABLE_MODELS})


def get_dataset_summary():
    """Returns dataset summary with min/max values and contribution values from .env."""
    df = pd.read_csv("Housing.csv")
    summary = {}
    # Load environment variables from .env file
    load_dotenv()

    for column in df.columns:
        unique_values = df[column].dropna().unique()

        # **For Numeric Columns → Min, Max, and Contribution**
        if df[column].dtype in ["int64", "float64"]:
            contribution_key = f"{column.upper()}_CONTRIBUTION"
            summary[column] = {
                "min": int(df[column].min()),
                "max": int(df[column].max()),
                "contribution": int(os.getenv(contribution_key, 0))  # Default to 0 if missing
            }

        # **For Yes/No Binary Columns → ["yes", "no"] and Contribution**
        elif all(str(val).strip().lower() in ["yes", "no"] for val in unique_values):
            contribution_key = f"{column.upper()}_CONTRIBUTION"
            summary[column] = {
                "values": ["yes", "no"],
                "contribution": int(os.getenv(contribution_key, 0))  # Default to 0 if missing
            }

        # **For Categorical/Text Columns → Unique Values and Contributions**
        else:
            contribution_data = {}
            for value in unique_values:
                contribution_key = f"{column.upper()}_{str(value).replace('-', '_').upper()}_CONTRIBUTION"
                contribution_data[str(value)] = int(os.getenv(contribution_key, 0))  # Default to 0

            summary[column] = {
                "values": list(map(str, unique_values)),
                "contributions": contribution_data
            }

    return summary

@app.route("/")
def home():
    dataset_summary = get_dataset_summary()
    return render_template("index.html", dataset_summary=dataset_summary, models=AVAILABLE_MODELS)

@app.route("/get-dataset-summary", methods=["GET"])
def dataset_summary():
    return jsonify(get_dataset_summary())

@app.route("/update-env", methods=["POST"])
def update_env():
    """API route to update `.env` file with new contribution values."""
    try:
        data = request.json
        dotenv_path = ".env"
        env_values = dotenv_values(dotenv_path)  # Load existing values

        # Update `.env` file with new values
        with open(dotenv_path, "w") as env_file:
            for key, value in data.items():
                key = key.replace("contribution-","").upper() + "_CONTRIBUTION"
                env_values[key] = str(value)  # Ensure values are strings
            for key, value in env_values.items():
                env_file.write(f"{key}={value}\n")

        # Reload the updated environment variables
        load_dotenv(dotenv_path, override=True)

        return jsonify({"success": True, "message": "Environment values updated successfully."})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/preprocess-data", methods=["POST"])
def preprocess_data():
    """
    API to preprocess data. 
    Expects JSON body with 'price_classes' and 'house_classes' parameters.
    """
    try:
        data = request.get_json()
        num_price_classes = int(data.get("price_classes", 5))
        num_house_types = int(data.get("house_classes", 5))

        # Call preprocessing function
        preprocessing.preprocess_data(num_price_classes, num_house_types)

        return jsonify({"success": True})  # ✅ Return success message

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle any errors gracefully

def async_train_model(model_name, model_type, hyperparams):
    global training_results
    training_results[model_type] = models.train_model(model_name, model_type, hyperparams)

@app.route("/train-random-forest", methods=["POST"])
def train_random_forest_route():
    """Train Random Forest and return results (accuracy, confusion matrices, model paths)."""
    data = request.get_json()
    hyperparams = data.get("hyperparams", {})

    try:
        result = random_forest.train_random_forest(n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"], min_samples_split=hyperparams["min_samples_split"])
        return jsonify(result)  # ✅ Send training results as JSON
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500  # ❌ Return error if something goes wrong

@app.route("/get-confusion-matrix/<filename>", methods=["GET"])
def get_confusion_matrix(filename):
    """Fetch saved confusion matrix image."""
    file_path = f"results/{filename}"
    try:
        return send_file(file_path, mimetype='image/png')
    except Exception:
        return jsonify({"success": False, "error": "File not found"}), 404  # ❌ Return error if file is missing


@app.route("/get-training-result", methods=["GET"])
def get_training_result():
    model_type = request.args.get("type", "price")
    return jsonify(training_results.get(model_type, {"message": "Training not completed yet"}))

if __name__ == "__main__":
    training_results = {}
    app.run(debug=True)
