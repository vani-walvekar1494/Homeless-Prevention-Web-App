from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model and tokenizer for chat
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Store conversation history
conversation_history_ids = None

@app.route('/')
def index():
    return render_template('index.html')

# Individual login and form
@app.route('/individual_login')
def individual_login():
    return render_template('individual.html')

@app.route('/submit_individual', methods=['POST'])
def submit_individual():
    name = request.form['name']
    age = request.form['age']
    # Process additional form data...
    # Here, you can save the data to a database
    return redirect(url_for('index'))

# Guest login and form
@app.route('/guest_login')
def guest_login():
    return render_template('guest.html')

@app.route('/submit_guest', methods=['POST'])
def submit_guest():
    location = request.form['location']
    emergency = request.form['emergency']
    # Process additional form data...
    # Here, you can save the data to a database
    return redirect(url_for('index'))

# Donator login and form
@app.route('/donator_login')
def donator_login():
    return render_template('donator.html')

@app.route('/submit_donator', methods=['POST'])
def submit_donator():
    name = request.form['name']
    govt_id = request.form['govt_id']
    # Process additional form data...
    # Here, you can save the data to a database
    return redirect(url_for('index'))

# Admin login and dashboard
@app.route('/admin_login')
def admin_login():
    return render_template('admin.html')

# Admin management routes (stub examples)
@app.route('/manage_individual')
def manage_individual():
    return "Manage Individual Users"

@app.route('/verify_requests')
def verify_requests():
    return "Verify User Requests"

@app.route('/emergency_requests')
def emergency_requests():
    return "View Emergency Requests"

@app.route('/manage_shelters')
def manage_shelters():
    return "Manage Shelters"

# Chat functionality
@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history_ids

    # Get user input from the request
    user_input = request.json.get("message")

    # Encode the new user input, add the EOS token, and append to the conversation history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if conversation_history_ids is not None:
        # Append the new user input to the existing conversation history
        conversation_history_ids = torch.cat([conversation_history_ids, new_user_input_ids], dim=-1)
    else:
        conversation_history_ids = new_user_input_ids

    # Generate a response from the model
    bot_output_ids = model.generate(conversation_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Get the bot's reply and convert it back to text
    bot_reply = tokenizer.decode(bot_output_ids[:, conversation_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Return the bot's reply as JSON
    return jsonify({"reply": bot_reply})

if __name__ == '__main__':
    app.run(debug=True)
