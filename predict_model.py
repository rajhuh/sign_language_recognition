import cv2
import mediapipe as mp
import numpy as np
import joblib

# load trained model
model = joblib.load("sign_model.pkl")

# mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # handle camera failure
    if not ret:
        print("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # normalization (same as training)
            landmarks = landmarks - landmarks[0]
            max_val = np.max(np.abs(landmarks))

            if max_val != 0:
                landmarks = landmarks / max_val

            input_data = landmarks.flatten().reshape(1, -1)

            # prediction with safety
            try:
                prediction = model.predict(input_data)[0]
            except Exception:
                prediction = "Error"

    # display prediction
    cv2.putText(
        frame,
        f"Sign: {prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Detection", frame)

    # press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
