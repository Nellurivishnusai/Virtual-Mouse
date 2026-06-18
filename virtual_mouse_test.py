import cv2
import mediapipe as mp
import pyautogui
import math

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip (8)
            x1 = hand_landmarks.landmark[8].x
            y1 = hand_landmarks.landmark[8].y

            # Thumb tip (4)
            x2 = hand_landmarks.landmark[4].x
            y2 = hand_landmarks.landmark[4].y

            # Convert to screen coords
            screen_x = int(x1 * screen_w)
            screen_y = int(y1 * screen_h)

            # Move cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Distance between thumb & index
            distance = math.hypot(x2 - x1, y2 - y1)

            # Click when fingers close
            if distance < 0.03:
                pyautogui.click()

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
