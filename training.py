import cv2
import os

def create_dataset():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    person_name = input("Enter the person's name: ")
    user_image_folder = f'static/faces/{person_name}'
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)

    print(f"Starting to capture images for {person_name}. Press 'q' to stop.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            img_name = f"{person_name}_{count}.jpg"
            cv2.imwrite(os.path.join(user_image_folder, img_name), face)
            count += 1

        cv2.imshow('Creating Dataset', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Images captured for {person_name}: {count}")

if __name__ == '__main__':
    create_dataset()
