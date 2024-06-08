import cv2
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier(
    'Cascade-FD-model/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascade-FD-model/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascade-FD-model/haarcascade_smile.xml')

def detect_face_eyes_smiles(rgb_image, scale_factor, min_neighbors):
    # Converting the image to gray
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # this returns the coordinates of the faces as x, y, w, h
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

    # copy of the frame
    face_only = frame.copy()
    face_eyes = frame.copy()
    face_smiles = frame.copy()
    face_eyes_smiles = frame.copy()

    # Looping through the faces
    for (x, y, w, h) in faces:

        # Drawing a rectangle around the face
        cv2.rectangle(face_eyes, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_only, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_smiles, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_eyes_smiles, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Getting the region of interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color_eyes = face_eyes[y:y + h, x:x + w]
        roi_color_smiles = face_smiles[y:y + h, x:x + w]
        roi_color_eyes_smiles = face_eyes_smiles[y:y + h, x:x + w]

        # Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 15)

        # Looping through the eyes
        for (ex, ey, ew, eh) in eyes:
            # Drawing a rectangle around the eyes
            cv2.rectangle(roi_color_eyes_smiles, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.rectangle(roi_color_eyes, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detecting the smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)

        # Looping through the smiles
        for (sx, sy, sw, sh) in smiles:
            # Drawing a rectangle around the smile
            cv2.rectangle(roi_color_smiles, (sx, sy),
                          (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.rectangle(roi_color_eyes_smiles, (sx, sy),
                          (sx + sw, sy + sh), (0, 0, 255), 2)

    face_only = cv2.cvtColor(face_only, cv2.COLOR_BGR2RGB)
    face_eyes = cv2.cvtColor(face_eyes, cv2.COLOR_BGR2RGB)
    face_smiles = cv2.cvtColor(face_smiles, cv2.COLOR_BGR2RGB)
    face_eyes_smiles = cv2.cvtColor(face_eyes_smiles, cv2.COLOR_BGR2RGB)

    return face_only, face_eyes, face_smiles, face_eyes_smiles
