from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        age = float(request.form["age"])
        gender = request.form["gender"]
        education = request.form["education"]
        job_title = request.form["job_title"]
        experience = float(request.form["experience"])

        # Build DataFrame exactly like training features
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": experience
        }])

        # Pipeline handles all preprocessing + prediction
        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=f"₹ {prediction:,.2f}"  # nicely formatted
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)