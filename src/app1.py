import cv2
import os
import time
import threading
import pickle
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# ================= CONFIG =================
DATASET_DIR = "dataset"
MODEL_PATH = "face_model.pkl"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

capturing = False
capture_count = 0
MAX_CAPTURE = 20
current_label = ""

training = False
model = None
labels_map = {}

# ================= LOAD MODEL =================
def load_model():
    global model, labels_map
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model, labels_map = pickle.load(f)
        print("✅ Model loaded")
    else:
        model = None
        labels_map = {}
        print("⚠️ No model found")

load_model()

# ================= CAMERA STREAM =================
def gen_frames():
    global capturing, capture_count

    while True:
        with camera_lock:
            ret, frame = camera.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            if capturing and capture_count < MAX_CAPTURE:
                save_path = os.path.join(DATASET_DIR, current_label)
                os.makedirs(save_path, exist_ok=True)
                filename = f"{int(time.time()*1000)}.jpg"
                cv2.imwrite(os.path.join(save_path, filename), face_img)
                capture_count += 1
                time.sleep(0.15)

            label_text = "Unknown"
            if model is not None:
                face_resized = cv2.resize(face_img, (100, 100))
                pred = model.predict([face_resized.flatten()])[0]
                label_text = labels_map.get(pred, "Unknown")

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label_text, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if capturing and capture_count >= MAX_CAPTURE:
            capturing = False

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    global capturing, capture_count, current_label
    data = request.json
    current_label = data.get("name", "unknown")
    capture_count = 0
    capturing = True
    return jsonify({"status": "started"})

@app.route("/capture_status")
def capture_status():
    return jsonify({
        "capturing": capturing,
        "count": capture_count,
        "max": MAX_CAPTURE
    })

# ================= TRAIN MODEL =================
def train_worker():
    global training, model, labels_map

    X = []
    y = []
    label_id = 0
    labels_map = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        labels_map[label_id] = person
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100,100))
            X.append(img.flatten())
            y.append(label_id)
        label_id += 1

    if len(X) == 0:
        print("❌ No data to train")
        training = False
        return

    print("🧠 Training model...")
    from sklearn.svm import SVC
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, labels_map), f)

    model = clf
    training = False
    print("✅ Training done")

@app.route("/train", methods=["POST"])
def train():
    global training
    if training:
        return jsonify({"status": "busy"})

    training = True
    threading.Thread(target=train_worker, daemon=True).start()
    return jsonify({"status": "started"})

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
