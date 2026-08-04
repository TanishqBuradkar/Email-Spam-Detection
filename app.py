from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Fix path issue
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def predict_spam(message):
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    return "Spam" if prediction[0] == 1 else "Not Spam"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form.get("message")

        if message and message.strip():
            prediction = predict_spam(message)
        else:
            prediction = "Please enter a message"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)