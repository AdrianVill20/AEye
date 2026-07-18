import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

#buffer for text
UPD_INTERVAL = 0.5
last_upd_time = 0
display_text = ["Waiting for tracking..."]

# mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
  
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    
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
      
      #check time
      curr_time = time.time()
      if curr_time - last_upd_time > UPD_INTERVAL:
        last_upd_time = curr_time
        
        #coordinates of shoulders/wrists
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        #display text
        display_text = [
          f"Left Shoulder: {l_shoulder.x:.2f}, {l_shoulder.y:.2f}",
          f"Right Shoulder: {r_shoulder.x:.2f}, {r_shoulder.y:.2f}",
          f"Left Wrist: {l_wrist.x:.2f}, {l_wrist.y:.2f}",
          f"Right Wrist: {r_wrist.x:.2f}, {r_wrist.y:.2f}",
        ]
    except:
      pass
    
    # rendering detections 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                             mp_drawing.DrawingSpec(color = (245,66,230), thickness = 1, circle_radius = 3),
                                                             mp_drawing.DrawingSpec(color = (0, 0, 0), thickness = 1, circle_radius = 2)
                                                             )
    
    #new window for stats
    stats = np.full((130,225,3),255, dtype=np.uint8)
    
    y_start = 28
    for i, line in enumerate(display_text):
      y = y_start + (i * 30)
      cv2.putText(stats, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
      
    # camera and stat feeds
    cv2.imshow('AEye - Side Camera Feed', image)
    cv2.imshow('AEye - Live Stats', stats)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
 
cap.release()
cv2.destroyAllWindows()

