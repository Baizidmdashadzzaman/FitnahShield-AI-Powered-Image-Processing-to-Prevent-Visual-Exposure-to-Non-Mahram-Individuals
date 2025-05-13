import cv2
import numpy as np
import mediapipe as mp

# Gender model files
GENDER_MODEL = 'gender_net.caffemodel'
GENDER_PROTO = 'deploy_gender.prototxt'
GENDERS = ['Male', 'Female']

# Load OpenCV gender classification model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# Initialize MediaPipe Pose and Face Detection
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
pose = mp_pose.Pose()
face_detection = mp_face.FaceDetection(model_selection=0)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    pose_result = pose.process(rgb)

    # Detect face
    face_result = face_detection.process(rgb)

    # Draw pose landmarks
    if pose_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Gender prediction from face
    if face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)

            # Crop face region
            face_img = frame[y1:y1+bh, x1:x1+bw]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                             (78.426, 87.768, 114.896), swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDERS[gender_preds[0].argmax()]
            else:
                gender = 'Unknown'

            # Draw face box and label
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (255, 0, 255), 2)
            cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Full Body & Gender Detection (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
