import face_recognition
image = face_recognition.load_image_file("16122.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)