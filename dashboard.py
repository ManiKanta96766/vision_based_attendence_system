import streamlit as st
import cv2
import torch
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load registered faces
with open('Data/embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)
with open('Data/names.pkl', 'rb') as f:
    known_names = pickle.load(f)

known_embeddings = np.array(known_embeddings)
THRESHOLD = 0.6

# Helper Functions
def calculate_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def center_of_box(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Streamlit Page
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("🎯 Face Recognition Attendance System")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Live Recognition", "📄 Attendance Records", "📊 Evaluation Metrics"])

# ======================== TAB 1 - LIVE RECOGNITION ========================
with tab1:
    st.header("Live Face Recognition with Anti-Spoofing")
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")
    save_button = st.button("Save Attendance")
    
    frame_placeholder = st.empty()
    recognized_placeholder = st.empty()
    unknown_placeholder = st.empty()

    recognized = {}
    unknown_faces = []
    running = False

    if start_button:
        video = cv2.VideoCapture(0)
        last_center = None
        static_frames = 0
        STATIC_LIMIT = 10
        BLUR_THRESHOLD = 50
        running = True

    while running:
        ret, frame = video.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                # Anti-Spoofing Checks
                current_center = center_of_box(box)
                if last_center is not None:
                    movement = np.linalg.norm(np.array(current_center) - np.array(last_center))
                    if movement < 5:  # pixel movement threshold
                        static_frames += 1
                    else:
                        static_frames = 0
                last_center = current_center

                blur_val = calculate_blur(face_crop)

                if static_frames >= STATIC_LIMIT or blur_val < BLUR_THRESHOLD:
                    # Spoof detected
                    cv2.putText(frame, "Spoof Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue

                # Face Recognition
                try:
                    face_crop = cv2.resize(face_crop, (160, 160))
                    face_crop = torch.tensor(face_crop).permute(2, 0, 1).float().to(device) / 255.0

                    with torch.no_grad():
                        emb = resnet(face_crop.unsqueeze(0)).cpu().numpy()[0]

                    min_dist = float('inf')
                    identity = "Unknown"

                    for idx, ref_emb in enumerate(known_embeddings):
                        dist = cosine(emb, ref_emb)
                        if dist < THRESHOLD and dist < min_dist:
                            min_dist = dist
                            identity = known_names[idx]

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if identity != "Unknown":
                        recognized[identity] = timestamp
                    else:
                        unknown_faces.append(timestamp)

                    # Draw boxes
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, identity, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except:
                    continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        recognized_list = [f"✅ {name} at {time}" for name, time in recognized.items()]
        unknown_list = [f"❓ Unknown detected at {time}" for time in unknown_faces]

        recognized_placeholder.markdown("### Recognized:")
        recognized_placeholder.text("\n".join(recognized_list))

        unknown_placeholder.markdown("### Unknowns:")
        unknown_placeholder.text("\n".join(unknown_list))

        if stop_button:
            running = False
            video.release()
            break

    if save_button and recognized:
        date = datetime.now().strftime("%d-%m-%Y")
        filename = f"Attendance_{date}.csv"
        with open(filename, "w") as f:
            f.write("NAME,TIME\n")
            for name, time in recognized.items():
                f.write(f"{name},{time}\n")
        st.success(f"Attendance saved to {filename}")

# ======================== TAB 2 - ATTENDANCE RECORDS ========================
with tab2:
    st.header("Attendance Records 📄")

    # List attendance files
    files = [f for f in os.listdir() if f.startswith('Attendance_') and f.endswith('.csv')]
    selected_file = st.selectbox("Select a date:", files)

    if selected_file:
        df = pd.read_csv(selected_file)
        st.dataframe(df)

# ======================== TAB 3 - EVALUATION METRICS ========================
with tab3:
    st.header("Evaluation Metrics 📊")

    if st.button("Evaluate Model"):
        # Load embeddings and names again
        X_train, X_test, y_train, y_test = train_test_split(known_embeddings, known_names, test_size=0.3, random_state=42)

        y_true = []
        y_pred = []

        for i in range(len(X_test)):
            true_label = y_test[i]
            test_emb = X_test[i]

            min_dist = float('inf')
            predicted_label = "Unknown"

            for j in range(len(X_train)):
                dist = cosine(test_emb, X_train[j])
                if dist < THRESHOLD and dist < min_dist:
                    min_dist = dist
                    predicted_label = y_train[j]

            y_true.append(true_label)
            y_pred.append(predicted_label)

        y_true_binary = [1 if label != "Unknown" else 0 for label in y_true]
        y_pred_binary = [1 if label != "Unknown" else 0 for label in y_pred]

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true_binary, y_pred_binary)
        recall = recall_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)

        false_accepts = sum((np.array(y_true) != np.array(y_pred)) & (np.array(y_pred) != "Unknown"))
        false_rejects = sum((np.array(y_true) != np.array(y_pred)) & (np.array(y_pred) == "Unknown"))

        FAR = false_accepts / len(y_true) * 100
        FRR = false_rejects / len(y_true) * 100

        st.subheader("Results:")
        st.write(f"Accuracy: **{acc*100:.2f}%**")
        st.write(f"Precision: **{precision*100:.2f}%**")
        st.write(f"Recall: **{recall*100:.2f}%**")
        st.write(f"F1 Score: **{f1*100:.2f}%**")
        st.write(f"False Acceptance Rate (FAR): **{FAR:.2f}%**")
        st.write(f"False Rejection Rate (FRR): **{FRR:.2f}%**")
