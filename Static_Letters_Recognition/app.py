from flask import Flask, render_template, Response
import cv2
import pickle
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Labels dict to map model predictions to characters
labels_dict = {0: 'A', 1: 'B', 2: 'L', '0': 'A', '1': 'B', '2': 'L'}

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame (CRITICAL: must match inference_classifier.py)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # We must reset data for EVERY frame
            data_aux = []
            x_ = []
            y_ = []

            # Focus only on the first hand detected to match basic training
            hand_landmarks = results.multi_hand_landmarks[0] 

            # Step 1: Collect coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Step 2: Normalize (Make the model location-invariant)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Step 3: Check if data_aux length matches your training (usually 42)
            if len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    raw_result = prediction[0]
                    predicted_character = labels_dict.get(raw_result, str(raw_result))

                    # Draw visual feedback
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (233, 69, 96), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (78, 204, 163), 3, cv2.LINE_AA)
                except Exception as e:
                    print(f"Prediction error: {e}")

        # Stream the result
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    # Modern Dark UI Template
    return '''
    <html>
        <head>
            <title>AI Sign Language Interpreter</title>
            <style>
                body { 
                    font-family: 'Segoe UI', sans-serif; 
                    background-color: #1a1a2e; 
                    color: #ffffff; 
                    text-align: center; 
                    margin: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                }
                .container {
                    background: #16213e;
                    padding: 30px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    border: 1px solid #0f3460;
                }
                h1 { 
                    margin-bottom: 20px;
                    color: #e94560;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                }
                .video-box {
                    border: 6px solid #e94560;
                    border-radius: 15px;
                    overflow: hidden;
                    line-height: 0;
                }
                .footer {
                    margin-top: 20px;
                    color: #4ecca3;
                    font-weight: bold;
                    font-size: 1.1em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤟 Sign Language AI</h1>
                <div class="video-box">
                    <img src="/video_feed" width="700">
                </div>
                <div class="footer">Status: System Online - Detecting Real-Time</div>
            </div>
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)