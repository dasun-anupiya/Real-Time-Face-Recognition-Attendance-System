import cv2
import os
import csv
import time
import numpy as np

def create_dataset(name, course, batch):
    """
    Captures face data from the camera and saves it to a dataset while guiding the user.

    Args:
        name (str): The name of the person.
        course (str): The course name.
        batch (str): The batch name.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    user_dir = os.path.join("face_data", name)
    os.makedirs(user_dir, exist_ok=True)

    count = 0
    instructions = ["Look straight", "Turn head left", "Turn head right", "Look up", "Look down"]
    instruction_index = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            instruction_text = instructions[instruction_index]
            cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y + h, x:x + w]
            face_filename = os.path.join(user_dir, f"{count}.jpg")
            cv2.imwrite(face_filename, face_roi)
            count += 1
            cv2.putText(frame, str(count), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if count % 20 == 0 and instruction_index < len(instructions) - 1:
                instruction_index += 1  # Move to the next instruction every 20 captures
                time.sleep(2)  # Pause briefly to allow user to adjust position

        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save student info to CSV
    with open('student_info.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat("student_info.csv").st_size == 0:
            writer.writerow(['Name', 'Course', 'Batch', 'Face Data Path'])
        writer.writerow([name, course, batch, user_dir])

def main():
    """Main function to add a new user."""
    name = input("Enter name: ")
    course = input("Enter course: ")
    batch = input("Enter batch: ")

    print("Follow the on-screen instructions for scanning your face.")
    create_dataset(name, course, batch)
    print(f"Face data for {name} saved successfully.")

if __name__ == "__main__":
    if not os.path.exists("face_data"):
        os.makedirs("face_data")
    if not os.path.exists("student_info.csv"):
        with open('student_info.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Course', 'Batch', 'Face Data Path'])
    main()
