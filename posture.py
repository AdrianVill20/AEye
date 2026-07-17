import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
  
  while cap.isOpened():
    ret, frame = cap.read()
    
    # recoloring image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    #make detection
    results = pose.process(image)
    
    #recoloring back
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #extract landmarks
    try:
      landmarks = results.pose_landmarks.landmark
      print(landmarks)
    except:
      pass
    
    # rendering detections 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                             mp_drawing.DrawingSpec(color = (245,66,230), thickness = 1, circle_radius = 3),
                                                             mp_drawing.DrawingSpec(color = (249, 246, 238), thickness = 1, circle_radius = 2)
                                                             )
    
    cv2.imshow('Mediapipe Feed', image)
  
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
 
cap.release()
cv2.destroyAllWindows()

