import cv2 
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector() #to detect face
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def midpoint(p1,p2):
    
    return (int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))

font = cv2.FONT_HERSHEY_COMPLEX

def blink(points, landarks):
    left_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)#left eye left
    right_point = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)#left eye right
    center_top = midpoint(landmarks.part(points[1]), landmarks.part(points[2]))#middle up
    center_bottom = midpoint(landmarks.part(points[5]), landmarks.part(points[4]))#middle down

    #h_line = cv2.line(frame, left_point, right_point,(0,255,255),2)
    #v_line = cv2.line(frame, center_top, center_bottom, (0,255,255), 2)

    hor_line = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1])) # to check for len
    ver_line = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    #print(f'vertical length:{ver_line}\n horizontal length:{hor_line}')
    div = hor_line/ver_line

    return div

def blinking_eyes(left_div,right_div):
    divide_eye = (left_div + right_div)/ 2
    return divide_eye

def gaze_ratio(eye_marks, landmarks):
    left_region = np.array([
             (landmarks.part(eye_marks[0]).x, landmarks.part(eye_marks[0]).y),
             (landmarks.part(eye_marks[1]).x, landmarks.part(eye_marks[1]).y),
             (landmarks.part(eye_marks[2]).x, landmarks.part(eye_marks[2]).y),
             (landmarks.part(eye_marks[3]).x, landmarks.part(eye_marks[3]).y),
             (landmarks.part(eye_marks[4]).x, landmarks.part(eye_marks[4]).y),
             (landmarks.part(eye_marks[5]).x, landmarks.part(eye_marks[5]).y)], np.int32) 
         #print(left_region) #location for left eye region
      #   cv2.polylines(frame, [left_region], True, (0,255,0), 2)

    h,w,_ = frame.shape
    mask = np.zeros((h,w), np.uint8) # create black screen
    cv2.polylines(mask, [left_region], True, 255, 2)
    cv2.fillPoly(mask, [left_region], 255)#g8
    eyes = cv2.bitwise_and(gray,gray, mask= mask) #g9e patung2 ang frames
         
    min_x = min(left_region[:,0])
    max_x = max(left_region[:,0])
    min_y = min(left_region[:,1])
    max_y = max(left_region[:,1])

    #eye = frame[min_y: max_y, min_x: max_x]
    #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_eye = eyes[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255,cv2.THRESH_BINARY)

    height,width = threshold_eye.shape#1
    left_threshold = threshold_eye[0:height, 0:int(width/2)]#2 halfing both threshold left and right
    left_side_white = cv2.countNonZero(left_threshold)
    right_threshold = threshold_eye[0:height,int(width/2): width]#3
    right_side_white = cv2.countNonZero(right_threshold)
    
    if left_side_white  == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 0
    else:
      gaze_ratio = left_side_white/right_side_white #4

    return gaze_ratio

while True:
   _, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #simple ky mas makita ang lines sa gray

   faces =detector(gray)
   for face in faces:
      #    print(face)
      #    x,y = face.left(), face.top()
      #    x1, y1 = face.right(), face.bottom()
         # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,255), 2) ## creates a bounding box for the face

         landmarks = predictor(gray, face)
         #print(landmarks.part(36)) #to see the 36 point landmark direction
      #    x = landmarks.part(36).x
      #    y = landmarks.part(36).y
      
      # #    cv2.circle(frame, (x,y), 3, (0,255,0), 3)
         left_div = blink([36,37,38,39,40,41], landmarks)
         right_div = blink([42,43,44,45,46,47], landmarks)
      #detect blinking
         if right_div and left_div > 5.5: #blink
             cv2.putText(frame, 'BLINKING', (50,150), font,2, (0,255,0))
      #gaze Detection
         gaze_ratio_left = gaze_ratio([36,37,38,39,40,41], landmarks)
         gaze_ratio_right = gaze_ratio([42,43,44,45,46,47], landmarks)
         gaze = (gaze_ratio_right + gaze_ratio_left)/2

         cv2.putText(frame, str(gaze), (50,150), font, 2, (0,255,0), 2)
         if gaze < 1:
            cv2.putText(frame, "looking right", (50,100), font, 2, (0,255,0), 2)
         elif 1< gaze < 3:
             cv2.putText(frame, "looking center", (50,100), font, 2, (0,255,0), 2)
         else: 
             cv2.putText(frame, "looking left", (50,100), font, 2, (0,255,0), 2)
         # cv2.putText(frame, str(left_side_white), (50,100), font, 2, (0,255,0), 2)
         # cv2.putText(frame, str(right_side_white), (50,150), font, 2, (0,255,0), 2)
         
         # eye = cv2.resize(gray_eye, None, fx=5, fy=5)
         # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
         # cv2.imshow('threshold', threshold_eye)
         # cv2.imshow('left_threshold', left_threshold)
         # cv2.imshow('right_threshold', right_threshold) 
   #flipped = cv2.flip(frame,1)
   cv2.imshow("Frame", frame)

   key = cv2.waitKey(1)
   if key == 27:
      break
cap.release()
cv2.destroyAllWindows()
