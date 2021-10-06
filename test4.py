import cv2 
import imutils 
   
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
# Initializing Frontal face cascade using haar classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) 
   
while True: 
    # Reading the video stream 
    ret, image = cap.read() 
    if ret: 
        image = imutils.resize(image,  
                               width=min(400, image.shape[1])) 
   
        # Detecting all the regions  
        # in the Image that has a  
        # pedestrians inside it 
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05) 
        # Detecting all faces
        faces = face_cascade.detectMultiScale(image,1.3,5)  
   
        # Drawing the regions in the  
        # Image 
        for (x, y, w, h) in regions: 
            cv2.rectangle(image, (x, y), 
                          (x + w, y + h),  
                          (0, 0, 255), 2) 
        for (fx, fy, fw, fh) in faces:
            center = (fx+(fw//2), fy+(fh//2))
            radius = (((fw+fh)//2)//2)+2
            cv2.circle(image, center, radius, (0,255,0), 1)
        # Counting persons
        nb = len(regions)
        nf = len(faces)
        np = max(nb,nf)
        # Showing the output Image 
        cv2.rectangle(image, (3,3),(397,35), (255, 0, 0),2,)
        cv2.putText(image,  ' TOTAL PERSONS DETECTED: '+str(np), (6, 25), cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 0, 255), 1)
        cv2.imshow("Cam-0 (Press q/Q to quit)", image) 
        k = cv2.waitKey(1)
        if k==ord('q') or k==ord('Q'):
            break
    else: 
         break
  
cap.release() 
cv2.destroyAllWindows() 