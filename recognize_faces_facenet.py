import cv2
import torch
import pickle
import numpy as np
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from win32com.client import Dispatch
import os

def speak(text):
    try:
        speaker = Dispatch('SAPI.SpVoice')
        speaker.Speak(text)
    except:
        print(f"[SPEAK] {text}")

def calculate_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def center_of_box(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Load data
with open("Data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
with open("Data/names.pkl", "rb") as f:
    names = pickle.load(f)

embeddings = np.array(embeddings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
video = cv2.VideoCapture(0)

attendance = set()
THRESHOLD = 0.6

last_center = None
static_frames = 0
STATIC_LIMIT = 10  # How many static frames allowed
BLUR_THRESHOLD = 50  # Lower = blurrier image

while True:
    ret, frame = video.read()
    if not ret:
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]

            # Motion Check
            current_center = center_of_box(box)
            if last_center is not None:
                movement = np.linalg.norm(np.array(current_center) - np.array(last_center))
                if movement < 5:  # pixel threshold
                    static_frames += 1
                else:
                    static_frames = 0
            last_center = current_center

            # Blurriness Check
            blur_val = calculate_blur(face_crop)

            # If static too long or too blurry => Spoof suspected
            if static_frames >= STATIC_LIMIT or blur_val < BLUR_THRESHOLD:
                cv2.putText(frame, "Spoof Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                continue  # Skip recognition for spoofed faces

            # Proceed to recognition
            face_crop = frame[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (160, 160))  # FaceNet expects 160x160 input
            face_crop = torch.tensor(face_crop).permute(2, 0, 1).float().to(device) / 255.0  # Normalize

            with torch.no_grad():
                emb = resnet(face_crop.unsqueeze(0)).cpu().numpy()[0]

            min_dist = float('inf')
            identity = "Unknown"

            for i, ref_emb in enumerate(embeddings):
                dist = cosine(emb, ref_emb)
                if dist < THRESHOLD and dist < min_dist:
                    min_dist = dist
                    identity = names[i]

            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, identity, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Attendance
            if identity != "Unknown" and identity not in attendance:
                attendance.add(identity)
                timestamp = datetime.now().strftime("%H:%M:%S")
                date = datetime.now().strftime("%d-%m-%Y")
                filename = f"Attendance_{date}.csv"
                entry = f"{identity},{timestamp}\n"
                if not os.path.exists(filename):
                    with open(filename, "w") as f:
                        f.write("NAME,TIME\n")
                with open(filename, "a") as f:
                    f.write(entry)
                speak(f"Attendance marked for {identity}")
                print(f"[✓] Attendance marked for {identity} at {timestamp}")

    cv2.imshow("Face Recognition + Anti-Spoofing", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
