import cv2
import mediapipe as mp
import numpy as np


gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']


mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb)
    pose_results = pose.process(rgb)

    if face_results.detections:
        for detection in face_results.detections:

            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + box_w, w), min(y + box_h, h)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263, 87.7689, 114.8958), swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            if gender == "Male":

                face_blur = cv2.GaussianBlur(face_img, (51, 51), 30)
                frame[y1:y2, x1:x2] = face_blur


                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark
                    x_coords = [int(l.x * w) for l in lm if 0 <= l.x <= 1]
                    y_coords = [int(l.y * h) for l in lm if 0 <= l.y <= 1]
                    if x_coords and y_coords:
                        bx1 = max(min(x_coords) - 20, 0)
                        by1 = max(min(y_coords) - 20, 0)
                        bx2 = min(max(x_coords) + 20, w)
                        by2 = min(max(y_coords) + 20, h)
                        body_roi = frame[by1:by2, bx1:bx2]
                        if body_roi.size != 0:
                            body_blur = cv2.GaussianBlur(body_roi, (51, 51), 30)
                            frame[by1:by2, bx1:bx2] = body_blur


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Blur Male Face + Body", frame)
    if cv2.waitKey(5) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
