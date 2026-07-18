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

    h_line = cv2.line(frame, left_point, right_point,(0,255,255),2)
    v_line = cv2.line(frame, center_top, center_bottom, (0,255,255), 2)

    hor_line = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1])) # to check for len
    ver_line = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    print(f'vertical length:{ver_line}\n horizontal length:{hor_line}')
    div = hor_line/ver_line

    return div

def blinking_eyes(left_div,right_div):
    divide_eye = (left_div + right_div)/ 2
    return divide_eye

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

         # divide_eye = (left_div + right_div)/ 2
      #detect blinking
         if right_div and left_div > 5.5: #blink
             cv2.putText(frame, 'BLINKING', (50,150), font,2, (0,255,0))
      #gaze Detection         
   #flipped = cv2.flip(frame,1)
   cv2.imshow("Frame", frame)

   key = cv2.waitKey(1)
   if key == 27:
      break
cap.release()
cv2.destroyAllWindows()
