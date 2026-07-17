import cv2

cam1 = cv2.VideoCapture(0)  
cam2 = cv2.VideoCapture(1)  

if not cam1.isOpened() or not cam2.isOpened():
    print("cant open one of the cams")
    exit()

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        break

    cv2.imshow('Cam 1', frame1)
    cv2.imshow('Cam 2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()