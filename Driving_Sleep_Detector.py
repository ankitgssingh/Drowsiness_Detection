print('''
##########################################################################
                                              
##########################################################################
# DOCUMENT CONTROL
##########################################################################
# SCRIPT AUTHOR        - Ankit Singh
# SCRIPT TITLE         - Driving Sleep Detector 
# DESCRIPTION          - Automatic sleep detection while driving to prevent accidents
##########################################################################
''')

##########################################################################
# LIBRARIES & PACKAGES
##########################################################################
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
##########################################################################
# RUN COMMAND FOR COMMAND LINE
##########################################################################
#python Driving_Sleep_Detector.py --webcam webcam_index  [default = 0 i.e. inbuilt webcam]
#python Driving_Sleep_Detector.py [If you are using only built in webcam]
##########################################################################
#Files Involved in the code
# frontol face cascade = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"
# Dlib facial landmark predictor = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"



#Defining the eye aspect ratio
def eye_aspect_ratio(eye):
    #Calculating the vertical distances for eyes
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    #Calclating the horizontal distance between eyes
    C = dist.euclidean(eye[0], eye[3])
    #Calculating the ratio
    ear = (A + B) / (2.0 * C)
    #Returning the eye aspect ration value
    return ear

#Defining function to apply the 'eye_aspect_ration' function on the eyes using landmarks
def calculated_final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]    #appying for left eye
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]    # for right eye

    leftEye  = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    lefteyeEAR  = eye_aspect_ratio(leftEye)
    righteyeEAR = eye_aspect_ratio(rightEye)

    #Final EAR
    ear = (lefteyeEAR + righteyeEAR) / 2.0
    return (ear, leftEye, rightEye)


#Similar funcion to calculate lip distance
def lip_distance(shape):
    upper_lip = shape[50:53]  #Can change according to the landmarks image
    upper_lip = np.concatenate((upper_lip, shape[61:64]))

    lower_lip = shape[56:59]   #Can change according to the landmarks image
    lower_lip = np.concatenate((lower_lip, shape[65:68]))

    #Calculating the mean for both lips
    upperlip_mean = np.mean(upper_lip, axis=0)
    lowerlip_mean = np.mean(lower_lip, axis=0)

    distance = abs(upperlip_mean[1] - lowerlip_mean[1])
    return distance


#Creating the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="camera on system")   #Default is inbuilt webcam
args = vars(ap.parse_args())
    
#Setting up the constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 35
YAWN_THRESH = 17
COUNTER = 0
alarm_status = False

#Setting the landmarks cascade file and predictor 
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #can get different cascades online
#detect = dlib.get_frontal_face_detector()  #Can use this dlib face detector as well for more accuracy 
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# start the video stream thread
print("[INFO] starting video stream ...")
vs = VideoStream(src=args["webcam"]).start() 
time.sleep(1.0)

#After reading video stream, looping over it to capture frames
while True:
    #grabbing the frame i.e. static pictures from the live video
    frame = vs.read()
    frame = imutils.resize(frame, width=460)
    #converting the images to grapscale for efficiency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting faces and getting the co-ordinates
    rects = detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    #Looping overt the face detections co-ordinates
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        shape = predict(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #Appying the functions to get the ear for both eyes and combined
        eye = calculated_final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]
        #Applying lip function 
        distance = lip_distance(shape)
        #Computing convex hull for both eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull= cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #Doing the same for the lips
        lips = shape[48:60]
        cv2.drawContours(frame, [lips], -1, (0, 255, 0), 1)

        #Checking the eye aspect ration and lip ration with the constants
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            #If the eyes were closed for sufficient time
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT ! FEELING SLEEPY ", (10, 30),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 60),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
        else:
            alarm_status = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()


