import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pyttsx3
import subprocess

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
        print("Face recognition model trained and saved successfully.")
    else:
        print("No faces or labels found to train the model.")

def recognize_faces():
    model = joblib.load('face_recognition_model.pkl')
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Base directory for audio files
    audio_dir = r'C:\Users\hp\Downloads\NoteDetection\TrainingData\audio'

    # Declaring the training set
    training_set = [
        '10F.jpg', '10B.jpg', '20F.jpg', '200F.jpg','2001.jpg','2002.jpg','2003.jpg','2004.jpg', '200B.jpg', '1001.jpg', '1002.jpg', '1003.jpg',
        '1004.jpg', '1005.jpg', '1006.jpg', '500F.jpg', '501.jpg', '502.jpg', '503.jpg', '504.jpg',
        '505.jpg', '506.jpg', '500B.jpg', '5001.jpg', '5002.jpg', '5003.jpg', '5004.jpg', '5005.jpg',
        '506.jpg', '2000F.jpg', '20B.jpg', '100.jpg', '201.jpg', '202.jpg', '203.jpg', '204.jpg',
        '205.jpg', '206.jpg'
    ]

    # Mapping of notes to their audio files
    audio_mapping = {
        '10F': '10.wav', '10B': '10.wav',
        '20F': '20.wav', '20B': '20.wav','204': '20.wav','203': '20.wav','205': '20.wav','201': '20.wav','206': '20.wav','202': '20.wav',
        '100': '100.wav', '1001': '100.wav', '1002': '100.wav', '1003': '100.wav', '1004': '100.wav', '1005': '100.wav', '1006': '100.wav',
        '501': '50.wav','504': '50.wav','500F': '50.wav','502': '50.wav','502': '50.wav','503': '50.wav', '500': '500.wav','5001': '500.wav','5002': '500.wav','5003': '500.wav','5004': '500.wav',
        '2001':'200.wav','2002':'200.wav','2003':'200.wav','2004':'200.wav','200F':'200.wav','200B':'200.wav',
        '500': '500.wav',
        '2000F': '2000.wav', '2000B': '2000.wav'
        # Continue for all other notes as needed
    }

    print("Starting face recognition and currency detection. Press 'q' to quit, 'c' to capture an image, and 'd' to detect currency.")

    image_captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(gray[y:y+h, x:x+w], (50, 50))

        cv2.imshow('Face Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('c'):  # Press 'c' to capture an image
            # Capture an image and save it
            cv2.imwrite('testsample.jpg', frame)
            print("Image captured.")
            image_captured = True
        elif key == ord('d') and image_captured:  # Press 'd' to detect currency
            test_img = cv2.imread('testsample.jpg')
            orb =" "
