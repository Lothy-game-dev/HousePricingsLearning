from flask import Flask, render_template, request
import joblib
import gbc  # Import file chứa model training

app = Flask(__name__)

def load_accuracy():
    try:
        with open("static/accuracy.txt", "r") as f:
            return f.readlines()
    except FileNotFoundError:
        return ["Accuracy data not available. Please run training script."]

@app.route("/", methods=["GET", "POST"])
def home():
    n_estimators = 200
    learning_rate = 0.1
    max_depth = 5

    if request.method == "POST":
        n_estimators = int(request.form["n_estimators"])
        learning_rate = float(request.form["learning_rate"])
        max_depth = int(request.form["max_depth"])

        # Gọi hàm train model từ file gbc.py
        gbc.train_model(n_estimators, learning_rate, max_depth)

    # Load kết quả accuracy từ file
    accuracy_data = load_accuracy()
    
    return render_template("index.html", 
                           accuracy_price=accuracy_data[0], 
                           accuracy_type=accuracy_data[1],
                           image_price="confusion_matrix_price.png", 
                           image_type="confusion_matrix_type.png",
                           n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           max_depth=max_depth)

if __name__ == "__main__":
    app.run(debug=True)
