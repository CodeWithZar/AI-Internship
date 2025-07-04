import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

mask_classifier = load_model("mask_detector.h5")

webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    if not success:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_found = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_found:
        face_image = frame[y:y+h, x:x+w]

        processed_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        processed_face = cv2.resize(processed_face, (100, 100))
        processed_face = processed_face / 255.0
        processed_face = np.reshape(processed_face, (1, 100, 100, 1))

        prediction = mask_classifier.predict(processed_face)[0]
        label = "Wearing Mask" if prediction[0] > prediction[1] else "No Mask"
        color = (0, 255, 0) if label == "Wearing Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
