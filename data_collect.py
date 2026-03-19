import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ===== SETTINGS =====
SAMPLES_PER_CLASS = 300
CSV_FILE = "sign_data.csv"

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_label = None
count = 0
data = []

print("\nPress any letter key to start collecting")
print("Press ESC to exit\n")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if current_label and results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # ===== NORMALIZATION =====
            landmarks = landmarks - landmarks[0]
            max_val = np.max(np.abs(landmarks))

            if max_val != 0:
                landmarks = landmarks / max_val

            data.append(np.append(landmarks.flatten(), current_label))
            count += 1

            if count >= SAMPLES_PER_CLASS:
                df = pd.DataFrame(data)

                if os.path.exists(CSV_FILE):
                    df.to_csv(CSV_FILE, mode="a", header=False, index=False)
                else:
                    df.to_csv(CSV_FILE, index=False)

                print(f"✅ {current_label} saved")

                # RESET
                current_label = None
                count = 0
                data = []

    # ===== UI TEXT =====
    if current_label:
        cv2.putText(frame, f"Collecting : {current_label}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Samples : {count}/{SAMPLES_PER_CLASS}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        cv2.putText(frame, "Press a letter to start", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)

    cv2.imshow("Smart Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===== EXIT =====
    if key == 27:
        break

    # ===== START COLLECTING =====
    if key != 255 and chr(key).isalpha() and current_label is None:
        current_label = chr(key).upper()
        count = 0
        data = []
        print(f"\nCollecting for {current_label}")

cap.release()
cv2.destroyAllWindows()