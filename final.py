import cv2
import os
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_model():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    user_image_folder = r'C:\Users\hp\Desktop\final project\static\faces'  # Update this path as needed
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

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'english+f3')  # Set female voice

# Load the face recognition model
model = joblib.load('face_recognition_model.pkl')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create ORB detector for currency detection
orb = cv2.ORB_create()

# Assume these are your currency detection images
training_set = [os.path.join(r'C:\Users\hp\Desktop\final project', f) for f in os.listdir(r'C:\Users\hp\Desktop\final project')]
audio_mapping = {
    '10F': 'this is Ten rupee',
    '10A': 'this is Ten rupee',
    '10I': 'this is Ten rupee',
    '10J': 'this is Ten rupee',
    '10K': 'this is Ten rupee',
    '10L': 'this is Ten rupee',
    '10H': 'this is Ten rupee',
    '10G': 'this is Ten rupee',
    '10B': 'this is Ten rupee',
    '10C': 'this is Ten rupee',
    '10D': 'this is Ten rupee',
    '10E': 'this is Ten rupee',
    '101': 'this is Ten rupee',
    '102': 'this is Ten rupee',
    '103': 'this is Ten rupee',
    '104': 'this is Ten rupee',
    '20F': 'this is Twenty rupee',
    '200F': 'Two Hundred rupee',
    '2001': 'this is Two Hundred rupee',
    '2002': 'this is Two Hundred rupee',
    '2003': 'this is Two Hundred rupee',
    '2004': 'this is Two Hundred rupee',
    '200': 'this is Two Hundred rupee',
    '2001': 'this is Two Hundred rupee',
    '2005': 'this is Two Hundred rupee',
    '2004': 'this is Two Hundred rupee',
    '2006': 'this is Two Hundred rupee',
    '2007': 'this is Two Hundred rupee',
    '2008': 'this is Two Hundred rupee',
    '2009': 'this is Two Hundred rupee',
    '20010': 'this is Two Hundred rupee',
    '20011': 'this is Two Hundred rupee',
    '20012': 'this is Two Hundred rupee',
    '20013': 'this is Two Hundred rupee',
    '20014': 'this is Two Hundred rupee',
    '20015': 'this is Two Hundred rupee',
    '200B': 'Two Hundred rupee',
    '1001': 'this is Hundred rupee',
    '1002': 'this is Hundred rupee',
    '1003': 'this is Hundred rupee',
    '1004': 'this is Hundred rupee',
    '1005': 'this is Hundred rupee',
    '1006': 'this is Hundred rupee',
    '1007': 'this is Hundred rupee',
    '1008': 'this is Hundred rupee',
    '1009': 'this is Hundred rupee',
    '1010': 'this is Hundred rupee',
    '1011': 'this is Hundred rupee',
    '1012': 'this is Hundred rupee',
    '1013': 'this is Hundred rupee',
    '1014': 'this is Hundred rupee',
    '1015': 'this is Hundred rupee',
    '1016': 'this is Hundred rupee',
    '1017': 'this is Hundred rupee',
    '1018': 'this is Hundred rupee',
    '1019': 'this is Hundred rupee',
    '1020': 'this is Hundred rupee',
    '1021': 'this is Hundred rupee',
    '1022': 'this is Hundred rupee',
    '1023': 'this is Hundred rupee',
    '1024': 'this is Hundred rupee',
    '500F': 'this is Five Hundred rupee',
    '501': 'this Fifty rupee',
    '502': 'this Fifty rupee',
    '503': 'this Fifty rupee',
    '504': 'this Fifty rupee',
    '50B': 'this Fifty rupee',
    '50F': 'this Fifty rupee',
    '20F': 'this is Twenty rupee',
    '20A': 'this is Twenty rupee',
    '20I': 'this is Twenty rupee',
    '20J': 'this is Twenty rupee',
    '20K': 'this is Twenty rupee',
    '20L': 'this is Twenty rupee',
    '20H': 'this is Twenty rupee',
    '20G': 'this is Twenty rupee',
    '20B': 'this is Twenty rupee',
    '20C': 'this is Twenty rupee',
    '20D': 'this is Twenty rupee',
    '20E': 'this is Twenty rupee',
    '200A': 'Two Hundred rupee',
    '200B': 'Two Hundred rupee',
    '200C': 'Two Hundred rupee',
    '200D': 'Two Hundred rupee',
    '200E': 'Two Hundred rupee',
    '200F': 'Two Hundred rupee',
    '200I': 'Two Hundred rupee',
    '505': 'this Fifty rupee',
    '506': 'this Fifty rupee',
    '500B': 'this is Five Hundred rupee',
    '5001': 'this is Five hundred rupee',
    '5002': 'this is Five hundred rupee',
    '5003': 'this is Five hundred rupee',
    '5004': 'this is Five hundred rupee',
    '5005': 'this is Five hundred rupee',
    '2000F': 'this is Two thousand rupee',
    '2000B': 'this is Two thousand rupee'
}

print("Starting the application. Press 'q' to quit, 'c' to capture for face recognition, 's' for currency detection.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow('Face and Currency Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        # Face recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            predicted_name = model.predict([face.ravel()])[0]
            if predicted_name:
                print(f"Hello {predicted_name}!")
                engine.say(f"Hello {predicted_name}!")
                engine.runAndWait()
    elif key == ord('s'):
        # Currency detection
        cv2.imwrite('testsample.jpg', frame)
        test_img = cv2.imread('testsample.jpg')
        (kp1, des1) = orb.detectAndCompute(test_img, None)
        max_val = 0
        detected_note = ''
        if des1 is not None:
            for img_name in training_set:
                train_img = cv2.imread(img_name)
                if train_img is None:
                    continue
                (kp2, des2) = orb.detectAndCompute(train_img, None)
                if des2 is None:
                    continue
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                all_matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in all_matches if m.distance < 0.75 * n.distance]
                if len(good) > max_val:
                    max_val = len(good)
                    detected_note = os.path.splitext(os.path.basename(img_name))[0]
            if detected_note:
                note_text = audio_mapping.get(detected_note, f"Detected: {detected_note}")
                print(note_text)
                engine.say(note_text)
                engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
