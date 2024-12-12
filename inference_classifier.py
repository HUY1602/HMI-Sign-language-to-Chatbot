
#PHIÊN BẢN 2
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# from collections import deque
# import tkinter as tk
# import requests
# import time

# # Load model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Video capture
# cap = cv2.VideoCapture(0)

# # Mediapipe initialization
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Labels dictionary
# labels_dict = {0: 'tôi', 1: 'tên', 2: "h", 3: 'u', 4: 'y', 5: 'miêu tả', 6: 'con chó', 7: 'ok'}

# # Store recognized characters
# recognized_characters = deque(maxlen=20)

# # Variables to track prediction stability
# last_prediction = None
# stability_start_time = None
# STABILITY_THRESHOLD = 2  # Seconds of stability required

# # API information
# api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=api_key"  # Replace with the actual Gemini API endpoint
# api_key = "AIzaSyCv-8PfWgq2hoYCVU_XZCcl8rIDU5iBOto"  # Replace with your actual API Key

# # Tkinter window to display recognized characters and chatbot response
# root = tk.Tk()
# root.title("Sign Language to Chatbot")

# # Text display for recognized characters
# text_display = tk.Text(root, height=10, width=50)
# text_display.pack()

# # Text display for chatbot response
# response_display = tk.Text(root, height=10, width=50, bg="lightyellow")
# response_display.pack()


# def update_text_display():
#     """Update the recognized characters in the Tkinter window."""
#     text_display.delete("1.0", tk.END)
#     text_display.insert(tk.END, " ".join(recognized_characters))
#     root.update()


# def update_response_display(response):
#     """Update the chatbot response in the Tkinter window."""
#     response_display.delete("1.0", tk.END)
#     response_display.insert(tk.END, response)
#     root.update()


# def call_gemini_api(question):
#     """Send the recognized sentence to the Gemini API and return the response."""
#     try:
#         # Prepare the payload
#         data = {"prompt": {"messages": [{"text": question}]}}
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

#         # Send the request
#         response = requests.post(api_url, json=data, headers=headers)

#         # Handle the response
#         if response.status_code == 200:
#             response_data = response.json()
#             return response_data.get("candidates", [{}])[0].get("content", "No response received.")
#         else:
#             return f"Error {response.status_code}: {response.text}"
#     except Exception as e:
#         return f"Exception: {str(e)}"


# def process_hand_sign(frame, hand_landmarks, H, W):
#     """Process hand landmarks to predict character."""
#     global last_prediction, stability_start_time

#     x_ = [lm.x for lm in hand_landmarks.landmark]
#     y_ = [lm.y for lm in hand_landmarks.landmark]

#     data_aux = [(lm.x - min(x_), lm.y - min(y_)) for lm in hand_landmarks.landmark]
#     data_aux_flat = [coord for pair in data_aux for coord in pair]

#     if len(data_aux_flat) == 42:  # Ensure the number of features is correct
#         prediction = model.predict([np.asarray(data_aux_flat)])
#         predicted_character = labels_dict[int(prediction[0])]

#         # Stability logic
#         if predicted_character == last_prediction:
#             if stability_start_time is None:
#                 stability_start_time = time.time()
#             elif time.time() - stability_start_time >= STABILITY_THRESHOLD:
#                 recognized_characters.append(predicted_character)
#                 update_text_display()
#                 stability_start_time = None  # Reset stability timer

#                 # Send to Gemini if "ok" is detected
#                 if predicted_character == "ok":
#                     sentence = " ".join(recognized_characters)
#                     chatbot_response = call_gemini_api(sentence)
#                     update_response_display(chatbot_response)
#                     recognized_characters.clear()  # Clear for new input
#         else:
#             last_prediction = predicted_character
#             stability_start_time = None  # Reset if prediction changes

#         # Draw bounding box and label
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) + 10
#         y2 = int(max(y_) * H) + 10
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style(),
#             )
#             process_hand_sign(frame, hand_landmarks, H, W)

#     cv2.imshow("Hand Sign Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# root.destroy()

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# from collections import deque
# import tkinter as tk
# import requests
# import time

# # Load model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Video capture
# cap = cv2.VideoCapture(0)

# # Mediapipe initialization
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Labels dictionary
# labels_dict = {0: 'tôi', 1: 'tên', 2: "h", 3: 'u', 4: 'y', 5: 'miêu tả', 6: 'con chó', 7: 'ok'}

# # Store recognized characters
# recognized_characters = deque(maxlen=20)

# # Variables to track prediction stability
# last_prediction = None
# stability_start_time = None
# STABILITY_THRESHOLD = 2  # Seconds of stability required

# # API information for Flask
# FLASK_API_URL = "http://127.0.0.1:5000/chat"  # Assuming Flask runs locally

# # Tkinter window to display recognized characters
# root = tk.Tk()
# root.title("Recognized Characters")
# text_display = tk.Text(root, height=10, width=50)
# text_display.pack()

# def update_text_display():
#     """Update the recognized characters in the Tkinter window."""
#     text_display.delete("1.0", tk.END)
#     text_display.insert(tk.END, " ".join(recognized_characters))
#     root.update()

# def send_to_chatbot(sentence):
#     """Send the recognized sentence to the Flask chatbot and get the response."""
#     try:
#         response = requests.post(FLASK_API_URL, json={"message": sentence})
#         if response.status_code == 200:
#             return response.json().get('response', "No response from chatbot.")
#         else:
#             return f"Error: {response.status_code}"
#     except Exception as e:
#         return f"Exception: {str(e)}"

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 hand_landmarks, 
#                 mp_hands.HAND_CONNECTIONS, 
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             x_ = [lm.x for lm in hand_landmarks.landmark]
#             y_ = [lm.y for lm in hand_landmarks.landmark]

#             data_aux = [(lm.x - min(x_), lm.y - min(y_)) for lm in hand_landmarks.landmark]
#             data_aux_flat = [coord for pair in data_aux for coord in pair]

#             if len(data_aux_flat) == 42:  # Ensure the number of features is correct
#                 prediction = model.predict([np.asarray(data_aux_flat)])
#                 predicted_character = labels_dict[int(prediction[0])]

#                 # Check for stability
#                 if predicted_character == last_prediction:
#                     if stability_start_time is None:
#                         stability_start_time = time.time()
#                     elif time.time() - stability_start_time >= STABILITY_THRESHOLD:
#                         recognized_characters.append(predicted_character)
#                         update_text_display()
#                         stability_start_time = None  # Reset stability timer

#                         # Send the sentence to Flask chatbot when "ok" is detected
#                         if predicted_character == "ok":  # Trigger when "ok" is recognized
#                             sentence = " ".join(recognized_characters)
#                             chatbot_response = send_to_chatbot(sentence)
#                             print(f"Chatbot response: {chatbot_response}")
#                             recognized_characters.clear()  # Clear for new input
#                 else:
#                     last_prediction = predicted_character
#                     stability_start_time = None  # Reset if prediction changes

#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10
#                 x2 = int(max(x_) * W) + 10
#                 y2 = int(max(y_) * H) + 10

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     cv2.imshow('Hand Sign Detection', frame)

#     # Exit condition
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# root.destroy()

import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
import requests
import time

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Video capture
cap = cv2.VideoCapture(0)

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'tôi', 1: 'tên', 2: "h", 3: 'u', 4: 'y', 5: 'miêu tả', 6: 'con chó', 7: 'ok'}

# Store recognized characters
recognized_characters = deque(maxlen=20)

# Variables to track prediction stability
last_prediction = None
stability_start_time = None
STABILITY_THRESHOLD = 2  # Seconds of stability required

# API information for Flask
FLASK_API_URL = "http://127.0.0.1:5000/chat"  # Assuming Flask runs locally

# Tkinter window to display recognized characters
root = tk.Tk()
root.title("Recognized Characters")

# Text display for recognized characters
text_display = tk.Text(root, height=10, width=50)
text_display.pack()

# Text display for chatbot response
response_display = tk.Text(root, height=10, width=50, bg="lightyellow")
response_display.pack()

def update_text_display():
    """Update the recognized characters in the Tkinter window."""
    text_display.delete("1.0", tk.END)
    text_display.insert(tk.END, " ".join(recognized_characters))
    root.update()

def update_response_display(response):
    """Update the chatbot response in the Tkinter window."""
    response_display.delete("1.0", tk.END)
    response_display.insert(tk.END, response)
    root.update()

def send_to_chatbot(sentence):
    """Send the recognized sentence to the Flask chatbot and get the response."""
    try:
        response = requests.post(FLASK_API_URL, json={"message": sentence})
        if response.status_code == 200:
            chatbot_response = response.json().get('response', "No response from chatbot.")
            update_response_display(chatbot_response)  # Update response in Tkinter
            return chatbot_response
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = [(lm.x - min(x_), lm.y - min(y_)) for lm in hand_landmarks.landmark]
            data_aux_flat = [coord for pair in data_aux for coord in pair]

            if len(data_aux_flat) == 42:  # Ensure the number of features is correct
                prediction = model.predict([np.asarray(data_aux_flat)])
                predicted_character = labels_dict[int(prediction[0])]

                # Check for stability
                if predicted_character == last_prediction:
                    if stability_start_time is None:
                        stability_start_time = time.time()
                    elif time.time() - stability_start_time >= STABILITY_THRESHOLD:
                        recognized_characters.append(predicted_character)
                        update_text_display()
                        stability_start_time = None  # Reset stability timer

                        # Send the sentence to Flask chatbot when "ok" is detected
                        if predicted_character == "ok":  # Trigger when "ok" is recognized
                            sentence = " ".join(recognized_characters)
                            chatbot_response = send_to_chatbot(sentence)
                            print(f"Chatbot response: {chatbot_response}")
                            recognized_characters.clear()  # Clear for new input
                else:
                    last_prediction = predicted_character
                    stability_start_time = None  # Reset if prediction changes

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Sign Detection', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
