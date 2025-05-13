import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose and Face Detection
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose and face detection
    pose_results = pose.process(rgb)
    face_results = face_detection.process(rgb)

    # Blur face(s)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)

            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)

            # Extract and blur face region
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                face_blur = cv2.GaussianBlur(face_roi, (51, 51), 30)
                frame[y1:y2, x1:x2] = face_blur

    # Blur full body
    if pose_results.pose_landmarks:
        # Extract landmark coordinates
        landmarks = pose_results.pose_landmarks.landmark
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]

        # Calculate bounding box
        x_min = max(min(x_coords) - 20, 0)
        x_max = min(max(x_coords) + 20, w)
        y_min = max(min(y_coords) - 20, 0)
        y_max = min(max(y_coords) + 20, h)

        # Extract and blur body region
        body_roi = frame[y_min:y_max, x_min:x_max]
        if body_roi.size != 0:
            body_blur = cv2.GaussianBlur(body_roi, (51, 51), 30)
            frame[y_min:y_max, x_min:x_max] = body_blur

    cv2.imshow('Blurred Face and Body', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
