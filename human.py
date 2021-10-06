import cv2
import numpy as np

# boby_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
boby_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
total = 0
while True:
    r, img = video.read()
    img = cv2.flip(img,1)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bodies = boby_cascade.detectMultiScale(gray_img, 1.3, 5)
    # print(len(bodies))
    total += len(bodies)
    for (x,y,w,h) in bodies:
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2)
    cv2.rectangle(img, (3,3),(200,35), (255, 0, 0),2,)
    cv2.putText(img,  ' TOTAL CROWD: '+str(total), (6, 25), cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 0, 255), 1)
    cv2.imshow('Camera-0 (Press Q/q to quit)', img)
    #cv2.imshow('Camera-0 gray (Press Q/q to quit)', gray_img)
    k = cv2.waitKey(30)
    if k==ord('q') or k==ord('Q'):
        break
video.release()
cv2.destroyAllWindows()