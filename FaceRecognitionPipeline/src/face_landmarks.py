import tensorflow as tf
import numpy as np
import dlib
import imutils
import os
import cv2
from FaceDetection.TinyFacesDetector import TinyFacesDetector

model_pkl="FaceDetection/weights.pkl"
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)

images = cv2.imread('/Users/krishrana/Desktop/jt.jpg') 
#gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
predictor = dlib.shape_predictor('FaceDetection/shape_predictor_68_face_landmarks.dat')

d_rects = tiny_faces_detector.detect(images,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
    #print(d_rects)
for d in d_rects:
	x,y,w,h = d[0],d[1],d[2],d[3]

d_rects=[dlib.rectangle(x, y, w, h)]
for ret in d_rects:
    # We will determine the facial landmarks for the face region, then 
    # can convert the facial landmark (x, y)-coordinates to a NumPy array 
    shape = predictor(images, box=ret)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
    return coords
shape=shape_to_np(shape)
print(shape[19])
print(shape[29])
print(shape[0])
print(shape[16])

face=images[y:h, x:w]
partial_face=images[shape[19][1]-int(shape[19][1]*0.5):shape[29][1]+int(shape[29][1]*0.1), shape[0][0]-int(shape[0][0]*0.3):shape[16][0]+int(shape[16][0]*0.2)]
cv2.imshow('', partial_face)
cv2.waitKey(0)
