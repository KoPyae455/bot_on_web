from flask import Flask, request, jsonify
from chatbot_api import generate_response
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
from utils.send_email import send_email

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend is running ✅"})

@app.route('/hire', methods=['POST'])
def hire():
    data = request.json
    message = f"Name: {data['name']}\nEmail: {data['email']}\n\nMessage:\n{data['message']}"
    send_email("📬 New Hire Request", message)
    return jsonify({"message": "Thanks! Your message was sent."})


@app.route('/')
def home():
    return app.send_static_file('index.html')


from ai_bot.run_inference import get_bot_reply

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_bot_reply(user_input)
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
