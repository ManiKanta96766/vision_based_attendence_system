import cv2
import torch
import pickle
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

video = cv2.VideoCapture(0)
name = input("Enter your name: ")
embeddings = []
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    face = mtcnn(frame)
    if face is not None:
        face = face.to(device)
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0)).cpu().numpy()
        embeddings.append(embedding[0])
        count += 1
        print(f"[INFO] Collected {count}/5 embeddings")

        # Display feedback
        frame_disp = frame.copy()
        cv2.putText(frame_disp, f"Collecting: {count}/5", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Registering Face", frame_disp)

    if cv2.waitKey(1) == ord('q') or count >= 5:
        break

video.release()
cv2.destroyAllWindows()

# Save embeddings and name
if not os.path.exists("Data"):
    os.makedirs("Data")

# Load existing data if any
if os.path.exists("Data/embeddings.pkl"):
    with open("Data/embeddings.pkl", "rb") as f:
        existing_embeddings = pickle.load(f)
    with open("Data/names.pkl", "rb") as f:
        existing_names = pickle.load(f)
else:
    existing_embeddings, existing_names = [], []

# Add new data
existing_embeddings.extend(embeddings)
existing_names.extend([name] * len(embeddings))

# Save
with open("Data/embeddings.pkl", "wb") as f:
    pickle.dump(existing_embeddings, f)
with open("Data/names.pkl", "wb") as f:
    pickle.dump(existing_names, f)

print(f"[✅] Saved {len(embeddings)} embeddings for {name}.")
