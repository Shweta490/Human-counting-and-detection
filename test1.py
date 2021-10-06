import cv2
import imutils

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# boby_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
boby_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while True:
    r, img = video.read()
    img = cv2.flip(img,1)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    new_img = imutils.resize(img, width=min(400, img.shape[1]))
    (regions, _) = hog.detectMultiScale(new_img, winStride=(4, 4), padding=(4, 4), scale=1.05) 
    for (x, y, w, h) in regions: 
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Camera-0 (Press Q/q to quit)', img)
    # cv2.imshow('Camera-0 gray (Press Q/q to quit)', gray_img)
    k = cv2.waitKey(30)
    if k==ord('q') or k==ord('Q'):
        break
video.release()
cv2.destroyAllWindows()