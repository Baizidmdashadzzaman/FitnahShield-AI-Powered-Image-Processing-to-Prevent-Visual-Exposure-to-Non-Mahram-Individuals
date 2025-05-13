import cv2
import mediapipe as mp
import numpy as np

# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + box_w)
            y2 = min(h, y + box_h)

            face_img = frame[y1:y2, x1:x2]

            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # Blur only male faces
                if gender == 'Male':
                    blur_face = cv2.GaussianBlur(face_img, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blur_face

                # Optionally draw bounding box and label
                label = f'{gender}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Male Face Blur', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
