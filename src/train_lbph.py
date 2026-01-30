import cv2, os, json
import numpy as np

DATASET = "dataset"
TRAINER = "trainer/lbph.yml"
CASCADE = "haarcascade_frontalface_default.xml"

os.makedirs("trainer", exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE)
recognizer = cv2.face.LBPHFaceRecognizer_create()

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
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in detected:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("NO_FACE")
    exit(2)

recognizer.train(faces, np.array(labels))
recognizer.save(TRAINER)

with open("trainer/labels.json", "w") as f:
    json.dump(label_map, f)

print("OK")
