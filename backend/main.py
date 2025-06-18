from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.send_email import send_email
from chatbot_api import generate_response

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check route."""
    return jsonify({"message": "Flask backend is running ✅"})


@app.route("/")
def index():
    """Serve the frontend."""
    return app.send_static_file("index.html")


@app.route("/hire", methods=["POST"])
def hire():
    data = request.json
    message = (
        f"Name: {data['name']}\nEmail: {data['email']}\n\nMessage:\n{data['message']}"
    )
    send_email("📬 New Hire Request", message)
    return jsonify({"message": "Thanks! Your message was sent."})


@app.route("/chat", methods=["POST"])
def chat():
    """Return chatbot response for the given user message."""
    user_input = request.json.get("message")
    response = generate_response(user_input)
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
