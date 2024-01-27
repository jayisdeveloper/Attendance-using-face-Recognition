import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime, date, timedelta
from plyer import notification 

def load_images_and_encodings(folder_path):
    images = [] 
    image_names = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        images.append(encoding)
        image_names.append(os.path.splitext(filename)[0])
    return images, image_names

def mark_attendance(name, date_str):
    filename = f"attendance_{date_str}.csv"

    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write("Name,Timestamp\n")

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  
            if name in line:
                return  

    with open(filename, "a") as f:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{timestamp}\n")

def send_alert():
    notification.notify(
        title='Unrecognized Face Detected',
        message='An unrecognized face has been detected!',
        app_icon=None,  
        timeout=10  
    )

def check_absence(name):
    filename = f"attendance_{date_str}.csv"
    absences_count =  0

    for i in range(5):
        previous_date = current_date - timedelta(days=i + 1)
        previous_date_str = previous_date.strftime("%Y-%m-%d")
        previous_filename = f"attendance_{previous_date_str}.csv"

        if os.path.isfile(previous_filename):
            with open(previous_filename, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if name in line:
                        absences_count = 0
                        break
                else:
                    absences_count += 1

    if absences_count == 5:
        notification.notify(
            title='Attendance Alert',
            message=f'{name} has been absent for 5 consecutive days. Please come to the office.',
            app_icon=None,  
            timeout=10  
        )

known_face_encodings, known_face_names = load_images_and_encodings("known_faces")

cap = cv2.VideoCapture(0)

current_date = date.today()
date_str = current_date.strftime("%Y-%m-%d")

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjust the resizing factor as needed
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                mark_attendance(name, date_str)
                check_absence(name)

            else:
                send_alert()

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2  
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()   
cv2.destroyAllWindows()
