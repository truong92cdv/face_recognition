from flask import Flask, render_template, Response, request, jsonify
import cv2, os, time, threading
import numpy as np

app = Flask(__name__)

CAMERA_ID = 0
DATASET = "dataset"
TRAINER = "trainer/lbph.yml"
CASCADE = "haarcascade_frontalface_default.xml"

AUTO_CAPTURE_TOTAL = 50

os.makedirs(DATASET, exist_ok=True)
os.makedirs("trainer", exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE)
cap = cv2.VideoCapture(CAMERA_ID)

latest_frame = None
lock = threading.Lock()

capture_running = False
capture_count = 0
current_user = ""

recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = {}
model_ready = False


# ================= CAMERA THREAD =================
def camera_reader():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame.copy()
        time.sleep(0.01)


threading.Thread(target=camera_reader, daemon=True).start()


# ================= STREAM =================
def gen_frames():
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        if model_ready:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
                name = label_map.get(id_, "Unknown") if conf < 80 else "Unknown"

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{name}", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ================= CAPTURE =================
def capture_thread():
    global capture_running, capture_count

    capture_running = True
    capture_count = 0

    user_dir = os.path.join(DATASET, current_user)
    os.makedirs(user_dir, exist_ok=True)

    while capture_count < AUTO_CAPTURE_TOTAL:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        capture_count += 1
        cv2.imwrite(f"{user_dir}/{capture_count}.jpg", frame)

        time.sleep(0.15)  # rất quan trọng

    capture_running = False


@app.route("/capture", methods=["POST"])
def capture():
    global current_user
    if capture_running:
        return jsonify({"status": "busy"})

    current_user = request.json["name"]
    threading.Thread(target=capture_thread, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/capture_status")
def capture_status():
    return jsonify({
        "running": capture_running,
        "count": capture_count
    })


# ================= TRAIN =================
@app.route("/train", methods=["POST"])
def train():
    global recognizer, label_map, model_ready

    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for user in os.listdir(DATASET):
        user_dir = os.path.join(DATASET, user)
        if not os.path.isdir(user_dir):
            continue

        label_map[label_id] = user

        for img_name in os.listdir(user_dir):
            img = cv2.imread(os.path.join(user_dir, img_name))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(label_id)

        label_id += 1

    if len(faces) == 0:
        return jsonify({"status": "no_faces"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER)

    model_ready = True
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
