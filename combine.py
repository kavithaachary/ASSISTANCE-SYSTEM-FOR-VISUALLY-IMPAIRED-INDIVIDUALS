import cv2
import pyttsx3
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

print(cv2.__version__)


def train_model():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    user_image_folder = r'C:\Users\hp\Desktop\final project\static\faces'
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

train_model()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create ORB detector
orb = cv2.ORB_create()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'english+f3')  # Set female voice

# Declaring the training set
training_set = [
    '10F.jpg','10A.jpg','10I.jpg','10J.jpg', '10K.jpg','10L.jpg','10H.jpg','10G.jpg','10B.jpg','10C.jpg','10D.jpg','10E.jpg', '101.jpg', '102.jpg', '103.jpg', '104.jpg', '20F.jpg', '200F.jpg', '2001.jpg', '2002.jpg', '2003.jpg', '2004.jpg','200.jpg','2001.jpg','2005.jpg','2004.jpg','2006.jpg','2007.jpg','2008.jpg','2009.jpg','20010.jpg','20011.jpg','20012.jpg','20013.jpg','20014.jpg','20015.jpg', 
    '200B.jpg', '1001.jpg', '1002.jpg', '1003.jpg', '1004.jpg', '1005.jpg', '1006.jpg','1007.jpg','1008.jpg','1009.jpg','1010.jpg','1011.jpg','1012.jpg','1013.jpg','1014.jpg','1015.jpg', '1016.jpg','1017.jpg','1018.jpg','1019.jpg','1020.jpg','1021.jpg','1022.jpg','1023.jpg','1024.jpg','500F.jpg', '501.jpg', '502.jpg', '503.jpg', '504.jpg', '50B.jpg','50F.jpg','20F.jpg','20A.jpg','20I.jpg','20J.jpg', '20K.jpg','20L.jpg','20H.jpg','20G.jpg','20B.jpg','20C.jpg','20D.jpg','20E.jpg','200A.jpg','200B.jpg','200C.jpg','200D.jpg','200E.jpg','200F.jpg','200I.jpg',
    '505.jpg', '506.jpg', '500B.jpg', '5001.jpg', '5002.jpg', '5003.jpg', '5004.jpg', '5005.jpg','500A.jpg','500B.jpg','500C.jpg','500D.jpg','500E.jpg','500F.jpg','500G.jpg', '506.jpg', '2000F.jpg', '20B.jpg', '100.jpg', 
    '201.jpg', '202.jpg', '203.jpg', '204.jpg', '205.jpg', '206.jpg'
]

# Mapping of notes to their text labels for text-to-speech
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
    '500A': 'this is Five Hundred rupee',
    '500B': 'this is Five Hundred rupee',
    '500C': 'this is Five Hundred rupee',
    '500D': 'this is Five Hundred rupee',
    '500E': 'this is Five Hundred rupee',
    '500F': 'this is Five Hundred rupee',
    '500G': 'this is Five Hundred rupee',
    '501': 'this is Fifty rupee',
    '502': 'this is Fifty rupee',
    '503': 'this is Fifty rupee',
    '504': 'this is Fifty rupee',
    '50B': 'this is Fifty rupee',
    '50F': 'this is Fifty rupee',
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
    '200A': 'this is Two Hundred rupee',
    '200B': 'this is Two Hundred rupee',
    '200C': 'this is Two Hundred rupee',
    '200D': 'this is Two Hundred rupee',
    '200E': 'this is Two Hundred rupee',
    '200F': 'this is Two Hundred rupee',
    '200I': 'this is Two Hundred rupee',
    '505': 'this is this Fifty rupee',
    '506': 'this is this Fifty rupee',
    '500B': 'this is Five Hundred rupee',
    '5001': 'this is Five hundred rupee',
    '5002': 'this is Five hundred rupee',
    '5003': 'this is Five hundred rupee',
    '5004': 'this is Five hundred rupee',
    '5005': 'this is Five hundred rupee',
    '2000F': 'this is Two thousand rupee',
    '2000B': 'this is Two thousand rupee'
}


print("Press 's' to capture an image and detect the currency")
model = joblib.load('face_recognition_model.pkl')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image_captured = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    face=None 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = cv2.resize(gray[y:y+h, x:x+w], (50, 50))



    cv2.imshow('Currency Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' key to capture and detect
        cv2.imwrite('testsample.jpg', frame)
        test_img = cv2.imread('testsample.jpg')
        (kp1, des1) = orb.detectAndCompute(test_img, None)

        max_val = 8  # Reset max_val for each detection
        detected_note = ''

        if des1 is not None:
            for img_name in training_set:
                train_img = cv2.imread(img_name)
                if train_img is None:
                    print(f"Could not load {img_name}, skipping...")
                    continue
                (kp2, des2) = orb.detectAndCompute(train_img, None)

                if des2 is None:
                    continue

                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                all_matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for m, n in all_matches:
                    if m.distance < 0.789 * n.distance:
                        good.append([m])
                if len(good) > max_val:
                    max_val = len(good)
                    detected_note = img_name.split('.')[0]

            if detected_note:
                print(f'Detected: {detected_note} with {max_val} good matches')

                note_text = audio_mapping.get(detected_note)
                if note_text:
                    print(f"Speaking: {note_text}")
                    engine.say(note_text)
                    engine.runAndWait()

    elif key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):
            # Capture an image and save it
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured.")
            image_captured = True

            # If an image is captured, recognize faces in the captured image
        
            if image_captured:
                try:
                    predicted_name = model.predict([face.ravel()])[0]
                    if predicted_name:
                        engine.say(f"Hello {predicted_name}!")
                        engine.runAndWait()
                    else:
                        print("No face detected.")
                    image_captured = False  # Reset the flag after recognizing the captured image
                except Exception as exp:
                    pass 
cap.release()
cv2.destroyAllWindows()




