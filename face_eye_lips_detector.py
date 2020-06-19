import cv2

#importing cascades
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade=cv2.CascadeClassifier("haarcascade_smile.xml")
#Defining function for detection
"""takes as input the image in black and white (gray) and the original image (frame),
 and that will return the same image with the detector rectangles"""
def detect(grayscale,frame):
    faces=faceCascade.detectMultiScale(grayscale,1.3,5) #to locate one or several faces in the image
    #1.3=scalling factor, 5=min neighbour zones
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #a rectangle around the face
        #detecting eyes
        eye_gray=grayscale[y:y+h,x:x+w]
        eye_frames=frame[y:y+h,x:x+w]
        eyes=eyeCascade.detectMultiScale(eye_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eye_frames,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles=smileCascade.detectMultiScale(eye_gray,1.17,25)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(eye_frames,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
        
    return frame

video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    grayscale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #colour transformation
    canvas = detect(grayscale,frame)
    cv2.imshow('Video',canvas)
    #stop loop:
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break

video_capture.release() # We turn the webcam off.
del video_capture
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.