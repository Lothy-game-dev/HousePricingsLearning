from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import utils.preprocessing_data as preprocessing
import learn_models.random_forest as random_forest
import learn_models.gbc as gbc
import learn_models.logistic_regression as logistic_regression
import utils.generate_data as generate_data
import utils.preprocessing_predict_data as preprocessing_predict_data
import utils.data_prediction as data_prediction
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
        "solver": {"type": "str", "default": "lbfgs", "options": ["newton-cg", "lbfgs", "sag", "saga"], "description": "Algorithm to use in the optimization problem"}
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
    df = pd.read_csv("Housing_Augmented.csv")
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
                "avg": round(df[column].mean(), 2),  # Calculate and round the average value
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

        summary["price_class_count"] = int(os.getenv("PRICE_CLASS_COUNT", 5))
        summary["type_class_count"] = int(os.getenv("TYPE_CLASS_COUNT", 5))

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
                key = key.replace("contribution-","").replace("-", "_").upper() + "_CONTRIBUTION"
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
    
        dotenv_path = ".env"
        env_values = dotenv_values(dotenv_path)  # Load existing values

        # Update `.env` file with new values
        with open(dotenv_path, "w") as env_file:
            env_values["PRICE_CLASS_COUNT"] = str(num_price_classes)
            env_values["TYPE_CLASS_COUNT"] = str(num_house_types)
            for key, value in env_values.items():
                env_file.write(f"{key}={value}\n")

        # Call preprocessing function
        preprocessing.preprocess_data(num_price_classes=num_price_classes, num_house_types=num_house_types)

        return jsonify({"success": True})  # ✅ Return success message

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/train-model", methods=["POST"])
def train_model_route():
    """
    Combined endpoint for training a model.
    
    Expected JSON payload:
    {
        "model": "random_forest" | "gradient_boosting" | "logistic_regression",
        "hyperparams": { ... },
        "optimization_method": "grid" | "random" | "bayesian" | "none"
    }
    """
    data = request.get_json()
    model_name = data.get("model")
    hyperparams = data.get("hyperparams", {})
    opt_method = data.get("optimization_method", "none").lower()

    # Set optimization booleans based on the optimization_method value
    use_grid_search = (opt_method == "grid")
    use_random_search = (opt_method == "random")
    use_bayesian_optimization = (opt_method == "bayesian")

    try:
        if model_name == "random_forest":
            # Expected hyperparams: n_estimators, max_depth, min_samples_split
            result = random_forest.train_random_forest(
                n_estimators = hyperparams["n_estimators"],
                max_depth = hyperparams["max_depth"],
                min_samples_split = hyperparams["min_samples_split"],
                use_grid_search = use_grid_search,
                use_random_search = use_random_search,
                use_bayesian_optimization = use_bayesian_optimization
            )
        elif model_name == "gradient_boosting":
            # Expected hyperparams: n_estimators, learning_rate, max_depth
            result = gbc.train_gradient_boosting(
                n_estimators = hyperparams["n_estimators"],
                learning_rate = float(hyperparams["learning_rate"]),
                max_depth = hyperparams["max_depth"],
                use_grid_search = use_grid_search,
                use_random_search = use_random_search,
                use_bayesian_optimization = use_bayesian_optimization
            )
        elif model_name == "logistic_regression":
            # Expected hyperparams: max_iter, solver, C
            result = logistic_regression.train_logistic_regression(
                max_iter = hyperparams["max_iter"],
                solver = hyperparams["solver"],
                C = float(hyperparams["C"]),
                use_grid_search = use_grid_search,
                use_random_search = use_random_search,
                use_bayesian_optimization = use_bayesian_optimization
            )
        else:
            return jsonify({"success": False, "error": "Invalid model name provided."}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route("/predict-model", methods=["POST"])
def predict_model():
    data = request.get_json()
    model_name = data.get("model")
    prediction_type = data.get("prediction_type")
    n = int(data.get("n", 10))
    
    # Generate random CSV data
    csv_file, returnData = generate_data.generate_random_data_for_prediction(n=n)
    
    # Preprocess the generated data (will not split since file_name != 'Housing_Augmented.csv')
    preprocessed_file = preprocessing_predict_data.preprocess_predict_data(fileName=csv_file)
    
    predictions = data_prediction.predict_model(model_name, prediction_type, preprocessed_file)

    # Build a list of dictionaries, one per record.
    finalReturnData = []
    columns = [
        "area",
        "prefarea",
        "bedrooms",
        "bathrooms",
        "parking",
        "stories",
        "airconditioning",
        "basement",
        "guestroom",
        "mainroad",
        "hotwaterheating",
        "furnishingstatus"
    ]

    if not predictions:
        return jsonify({"success": False, "error": "Prediction failed."}), 500
    
    for i in range(n):
        record = {}
        for col in columns:
            # Assumes returnData[col] is indexable (e.g. a list or pandas Series)
            record[col] = returnData[col][i]
        if (prediction_type == "price"):
            record["price_level"] = "Price Level " + str(predictions[i])
        else:
            record["type_level"] = "Type Level " + str(predictions[i])
        finalReturnData.append(record)

    # Return a JSON response containing a list of plain dictionaries.
    return jsonify({"success": True, "predictions": finalReturnData, "csv_file": csv_file})

@app.route("/predict-model-file", methods=["POST"])
def predict_model_file():
    # Check if file is provided
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    # Save the uploaded file securely
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Retrieve additional parameters from the form
    model_name = request.form.get("model")
    prediction_type = request.form.get("prediction_type")

    # Preprocess the uploaded CSV file.
    # (This file should not contain the "price" column.)
    preprocessed_file = preprocessing_predict_data.preprocess_predict_data(fileName=file_path)

    # Get predictions using the preprocessed data.
    predictions = data_prediction.predict_model(model_name, prediction_type, preprocessed_file)
    if predictions is None:
        return jsonify({"success": False, "error": "Prediction failed."}), 500

    # Load the original CSV to extract the input data values.
    df_original = pd.read_csv(file_path)
    # Define the columns you want to include in the final output.
    columns = [
        "area",
        "prefarea",
        "bedrooms",
        "bathrooms",
        "parking",
        "stories",
        "airconditioning",
        "basement",
        "guestroom",
        "mainroad",
        "hotwaterheating",
        "furnishingstatus"
    ]
    finalReturnData = []

    # Build a record for each row combining selected feature values with the prediction.
    for i in range(len(predictions)):
        record = {}
        for col in columns:
            value = df_original[col].iloc[i]
            # Convert numpy types (e.g. numpy.int64) to Python int/float if necessary.
            if hasattr(value, "item"):
                record[col] = value.item()
            else:
                record[col] = value
        # Process the prediction value similarly.
        pred_val = predictions[i]
        if hasattr(pred_val, "item"):
            if (prediction_type == "price"):
                record["price_level"] = "Price Level " + str(pred_val.item())
            else:
                record["type_level"] = "Type Level " + str(pred_val)
        else:
            if (prediction_type == "price"):
                record["price_level"] = "Price Level " + str(pred_val)
            else:
                record["type_level"] = "Type Level " + str(pred_val)
        finalReturnData.append(record)

    return jsonify({"success": True, "predictions": finalReturnData})

@app.route("/get-confusion-matrix/<filename>", methods=["GET"])
def get_confusion_matrix(filename):
    """Fetch saved confusion matrix image."""
    file_path = f"results/{filename}"
    try:
        return send_file(file_path, mimetype='image/png')
    except Exception:
        return jsonify({"success": False, "error": "File not found"}), 404  # ❌ Return error if file is missing

@app.route("/get-training-log", methods=["GET"])
def get_training_log():
    log_file_path = os.path.join(os.getenv("LOG_DIR"), os.getenv("LOG_FILE"))
    if not os.path.exists(log_file_path):
        return jsonify({"success": False, "error": "Training log file not found"}), 404

    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()

    logs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        entry = {}
        parts = line.split(";")
        for part in parts:
            if ":" in part:
                key, val = part.split(":", 1)
                entry[key.strip()] = val.strip()
        logs.append(entry)

    return jsonify({"success": True, "logs": logs})

@app.route("/clear_logs", methods=["DELETE"])
def clear_logs():
    log_file_path = os.path.join(os.getenv("LOG_DIR"), os.getenv("LOG_FILE"))
    if os.path.exists(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("")
    
    cm_path = os.path.join(os.getenv("RESULTS_DIR"))
    for file in os.listdir(cm_path):
        os.remove(os.path.join(cm_path, file))

    model_path = os.path.join(os.getenv("MODEL_DIR"))
    for file in os.listdir(model_path):
        os.remove(os.path.join(model_path, file))

    return jsonify({"success": True})

if __name__ == "__main__":
    training_results = {}
    app.run(debug=True)
