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
         left_point = (landmarks.part(36).x, landmarks.part(36).y)#left eye left
         right_point = (landmarks.part(39).x, landmarks.part(39).y)#left eye right
         center_top = midpoint(landmarks.part(37), landmarks.part(38))#middle up
         center_bottom = midpoint(landmarks.part(41), landmarks.part(40))#middle down

         h_line = cv2.line(frame, left_point, right_point,(0,255,255),2)
         v_line = cv2.line(frame, center_top, center_bottom, (0,255,255), 2)

         hor_line = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1])) # to check for len # computes sqrt(dx² + dy²) — the straight-line (Euclidean) distance between two points.
         ver_line = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

         print(f'vertical length:{ver_line}\n horizontal length:{hor_line}')
         div = hor_line/ver_line
         if div > 5:
             cv2.putText(frame, 'BLINKING', (50,150), font,2, (0,255,0))         
   #flipped = cv2.flip(frame,1)
   cv2.imshow("Frame", frame)

   key = cv2.waitKey(1)
   if key == 27:
      break
cap.release()
cv2.destroyAllWindows()
