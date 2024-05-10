import cv2
import numpy as np
import joblib
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
import os

def train_model():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    user_image_folder = 'static/faces'
    userlist = os.listdir(user_image_folder)
    for user in userlist:
        user_folder = os.path.join(user_image_folder, user)
        for imgname in os.listdir(user_folder):
            img = cv2.imread(os.path.join(user_folder, imgname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user.split('_')[0])

    if faces and labels:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'face_recognition_model.pkl')
        print("Model trained and saved successfully.")
    else:
        print("No faces or labels found to train the model.")

def recognize_faces(threshold=0.5):
    model = joblib.load('face_recognition_model.pkl')
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    engine = pyttsx3.init()
    print("Starting face recognition. Press 'q' to quit, 'c' to capture and recognize faces.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Face Recognition', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
                face_array = face.ravel()
                probabilities = model.predict_proba([face_array])
                max_probability = np.max(probabilities)
                predicted_name = model.predict([face_array])[0] if max_probability >= threshold else "Unknown Person"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                message = f"Hello {predicted_name}!" if max_probability >= threshold else "Unknown person detected."
                engine.say(message)
                engine.runAndWait()

                print(message)  # Also print the message in the console

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_model()
    recognize_faces()
