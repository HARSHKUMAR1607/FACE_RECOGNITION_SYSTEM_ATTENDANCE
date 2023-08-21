import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

harsh_image = face_recognition.load_image_file("faces/harsh.jpeg")
harsh_encoding = face_recognition.face_encodings(harsh_image)[0]

Monika_image = face_recognition.load_image_file("faces/Monika.jpeg")
Monika_encoding = face_recognition.face_encodings(Monika_image)[0]

Tushar_image = face_recognition.load_image_file("faces/Tushar.jpg")
Tushar_encoding = face_recognition.face_encodings(Tushar_image)[0]

sourabh_image = face_recognition.load_image_file("faces/sourabh.jpeg")
sourabh_encoding = face_recognition.face_encodings(sourabh_image)[0]

known_face_encodings = [harsh_encoding, sourabh_encoding, Tushar_encoding, Monika_encoding]
known_face_names = ["Harsh", "Sourabh", "Tushar", "Monika"]

students = known_face_names.copy()

face_locations = []
face_encodings = [1]

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)


class OxFF:
    pass


while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)
        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_COMPLEX
                bottomLeftCornerofText = (10,100)
                fontscale = 1.5
                fontcolor = (255, 0, 0)
                thickness = 3
                linetype = 2
                cv2.putText(frame, name + " Present ", bottomLeftCornerofText,font, fontscale, fontcolor, thickness, linetype )

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M%S")
                    lnwriter.writerow([name, current_time])


    cv2.imshow("Attendace", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close












