import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime
import csv

def load_face_encodings(data_dir):
    """Loads face encodings and names from the dataset."""
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(data_dir):
        user_dir = os.path.join(data_dir, name)
        if os.path.isdir(user_dir):
            for filename in os.listdir(user_dir):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    try:
                        image = face_recognition.load_image_file(os.path.join(user_dir, filename))
                        face_encodings = face_recognition.face_encodings(image)
                        if len(face_encodings) > 0:
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(name)
                        else:
                            print(f"Warning: No face found in {os.path.join(user_dir, filename)}")
                    except Exception as e:
                        print(f"Error processing {os.path.join(user_dir, filename)}: {e}")
    return known_face_encodings, known_face_names

def record_attendance(name, attendance_file):
    """Records attendance in a CSV file."""
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file is empty
        if os.stat(attendance_file).st_size == 0:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, date_string, time_string])

def recognize_faces(known_face_encodings, known_face_names, attendance_file):
    """Recognizes faces from the camera and displays names on the screen."""
    cap = cv2.VideoCapture(0)
    attendance_marked = {} #Dictionary to store names that have already had attendance marked.

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in attendance_marked: # Check if attendance is already marked
                        record_attendance(name, attendance_file)
                        attendance_marked[name] = True # Mark attendance
                        print(f"Attendance marked for {name}") #Optional console output.

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if name in attendance_marked and attendance_marked[name]:
                 cv2.putText(frame, "Attendance Marked", (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run face recognition and display names."""
    data_dir = "face_data"
    attendance_file = "attendance.csv"

    if not os.path.exists(data_dir):
        print("Error: Face data directory not found.")
        return

    known_face_encodings, known_face_names = load_face_encodings(data_dir)
    if not known_face_encodings:
        print("Error: No face encodings found in the data directory.")
        return

    recognize_faces(known_face_encodings, known_face_names, attendance_file)

if __name__ == "__main__":
    main()