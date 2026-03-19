from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import numpy as np
import cv2
import joblib

app = Flask(__name__)

# load trained model
model = joblib.load("sign_model.pkl")

# mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

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
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"prediction": "-"})

        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # normalize same as training
        landmarks = landmarks - landmarks[0]
        max_val = np.max(np.abs(landmarks))

        if max_val != 0:
            landmarks = landmarks / max_val

        input_data = landmarks.flatten().reshape(1, -1)

        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": prediction})

    except Exception:
        return jsonify({"prediction": "Error"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)