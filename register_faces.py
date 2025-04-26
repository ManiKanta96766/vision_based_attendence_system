import cv2
import numpy as np
import pickle
import os
from facenet_pytorch import MTCNN

# Init camera and MTCNN
video = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=False)

facesdata = []
name = input("Enter your name: ")
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face, (50, 50))

            if count % 5 == 0:
                facesdata.append(resized_face)
            count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(len(facesdata)), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Register Face - MTCNN", frame)
    if cv2.waitKey(1) == ord('q') or len(facesdata) == 5:
        break

video.release()
cv2.destroyAllWindows()

# Save data
faces_data_np = np.asarray(facesdata).reshape(5, -1)

# Save names
if not os.path.exists('Data'):
    os.makedirs('Data')

if 'names.pkl' in os.listdir('Data'):
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * 5)
else:
    names = [name] * 5

with open('Data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

# Save face data
if 'faces_data.pkl' in os.listdir('Data'):
    with open('Data/faces_data.pkl', 'rb') as f:
        old_data = pickle.load(f)
    combined_data = np.append(old_data, faces_data_np, axis=0)
else:
    combined_data = faces_data_np

with open('Data/faces_data.pkl', 'wb') as f:
    pickle.dump(combined_data, f)

print("[✅] Face data saved using MTCNN.")
