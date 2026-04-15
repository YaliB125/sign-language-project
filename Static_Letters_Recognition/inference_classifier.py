import pickle
import mediapipe as mp
import numpy as np
import os
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

st.title("🤟 Sign Language Detector")
st.write("Click **START** below and show your hand to the camera to translate ASL letters in real-time.")

# Set up the WebRTC server configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            
            # Draw the skeleton on the hand
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract the 42 coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            # Predict if we have exactly 42 features
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = str(prediction[0])

    # Draw the predicted letter directly onto the video feed
    if predicted_character:
        cv2.putText(img, predicted_character, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

    # Return the modified frame back to the browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Start the Camera Stream ---
webrtc_streamer(
    key="sign-language",
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)