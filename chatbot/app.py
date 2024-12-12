from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# API URL and API Key for Gemini
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
API_KEY = "AIzaSyCv-8PfWgq2hoYCVU_XZCcl8rIDU5iBOto"  # Replace with your actual API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')  # Receive the message from the user

    # Prepare the payload for the Gemini API
    data = {
        "contents": [{
            "parts": [{"text": user_message}]
        }]
    }

    try:
        # Send the message to Gemini API
        response = requests.post(
            f'{GEMINI_API_URL}?key={API_KEY}', 
            json=data, 
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            response_data = response.json()
            bot_message = response_data['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"response": bot_message})
        else:
            return jsonify({"response": "Error: Unable to get a response from the API"}), 500

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

# import tkinter as tk
# from tkinter import messagebox
# import requests
# import time

# # API URL and API Key for Gemini
# GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
# API_KEY = "AIzaSyCv-8PfWgq2hoYCVU_XZCcl8rIDU5iBOto" 

# # Function to call the chatbot API with retry logic
# def call_chatbot_api(user_message):
#     data = {
#         "contents": [{
#             "parts": [{"text": user_message}]
#         }]
#     }

#     retry_attempts = 3  # Number of retries
#     for attempt in range(retry_attempts):
#         try:
#             response = requests.post(f'{GEMINI_API_URL}?key={API_KEY}', json=data, headers={'Content-Type': 'application/json'})
#             response_data = response.json()

#             if response.ok:
#                 return response_data['candidates'][0]['content']['parts'][0]['text']
#             else:
#                 return "Sorry, something went wrong."
#         except requests.exceptions.RequestException as e:
#             if attempt < retry_attempts - 1:
#                 time.sleep(2)  # Wait for 2 seconds before retrying
#                 continue
#             else:
#                 return f"Error: {str(e)}"

# # Function to send the message and get the response
# def on_send_button_click():
#     user_message = entry.get()  # Get user input
#     if user_message:
#         response = call_chatbot_api(user_message)  # Get the response from the chatbot API
#         display_response(response)  # Display the response in the window
#         entry.delete(0, tk.END)  # Clear the input field

# # Function to display the response in the Tkinter window
# def display_response(response):
#     response_label.config(text=f"Chatbot Response: {response}")

# # Set up the Tkinter window
# root = tk.Tk()
# root.title("Chatbot")

# # Set up the user input and button
# entry = tk.Entry(root, width=50)
# entry.pack(pady=20)

# send_button = tk.Button(root, text="Send", command=on_send_button_click)
# send_button.pack(pady=10)

# # Label for displaying the chatbot response
# response_label = tk.Label(root, text="Chatbot Response: ", width=50, height=4, relief="solid", anchor="w")
# response_label.pack(pady=20)

# # Run the Tkinter event loop
# root.mainloop()
