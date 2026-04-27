import cv2

# Open default camera (0)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to access camera")
        break

    # Display the camera feed
    cv2.imshow("Camera Test", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
