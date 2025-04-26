import numpy as np
import csv
import cv2
import os
from datetime import datetime
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from facenet_pytorch import MTCNN
from win32com.client import Dispatch

# Optional: speech
def speak(text):
    try:
        speaker = Dispatch('SAPI.SpVoice')
        speaker.Speak(text)
    except:
        print("[⚠️] Text-to-speech not supported on this system.")

# Load trained face encodings and names
with open('Data/names.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('Data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

print(f"[INFO] Faces data shape: {faces_data.shape}")

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, labels)

# Initialize webcam and MTCNN
video = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=False)

# CSV logging setup
col_names = ['NAME', 'TIME']

attendance_marked = set()

while True:
    ret, frame = video.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            try:
                resized_face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)
                prediction = knn.predict(resized_face)[0]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, prediction, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

                filename = f"Attendance_{date}.csv"
                record = [prediction, timestamp]

                if prediction not in attendance_marked:
                    attendance_marked.add(prediction)
                    if os.path.exists(filename):
                        with open(filename, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(record)
                    else:
                        with open(filename, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(col_names)
                            writer.writerow(record)

                    speak(f"Attendance marked for {prediction}")
                    print(f"[✓] Attendance marked for {prediction} at {timestamp}")

            except Exception as e:
                print(f"[Error] Failed to process face: {e}")

    cv2.imshow("Face Recognition Attendance (MTCNN)", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
