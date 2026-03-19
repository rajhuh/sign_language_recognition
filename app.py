from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("sign_model.pkl")

# MediaPipe hands - compatible with both old and new versions
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, static_image_mode=True)
    print("MediaPipe loaded via solutions API")
except AttributeError:
    # Newer mediapipe versions (0.11+) use Tasks API
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    hands = None
    print("MediaPipe solutions API not available - using Tasks API fallback")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"prediction": "-"})

        file = request.files["image"]
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"prediction": "-"})

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if hands is None:
            return jsonify({"prediction": "MediaPipe not available"})

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"prediction": "-"})

        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize same as training
        landmarks = landmarks - landmarks[0]
        max_val = np.max(np.abs(landmarks))
        if max_val != 0:
            landmarks = landmarks / max_val

        input_data = landmarks.flatten().reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"prediction": "Error"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
