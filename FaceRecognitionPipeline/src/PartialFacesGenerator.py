import tensorflow as tf
import numpy as np
import dlib
import imutils
import os
import cv2
from FaceDetection.TinyFacesDetector import TinyFacesDetector

model_pkl="FaceDetection/weights.pkl"
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
    return coords

def get_training_data(directory, op):
    for subdir in os.listdir(directory):
        path=os.path.join(directory, subdir)
        print(path)
        try:
            for img in os.listdir(path):
                try:
                    image_array=cv2.imread(os.path.join(path, img))
                    image_array=cv2.resize(image_array, (160,160))
                    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                    d_rects = tiny_faces_detector.detect(gray,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
                    for d in d_rects:
                        x,y,w,h = d[0],d[1],d[2],d[3]
                    face_rects=[dlib.rectangle(x, y, w, h)]
                    for ret in face_rects:
                        shape = predictor(gray, box=ret)
                    shape=shape_to_np(shape)
                    partial_face=image_array[shape[19][1]-int(shape[19][1]*0.5):shape[29][1]+int(shape[29][1]*0.1), 
                                             shape[0][0]-int(shape[0][0]*0.3):shape[16][0]+int(shape[16][0]*0.3)]
                    cv2.imwrite(os.path.join(op, subdir, img), partial_face)
                except Exception as e:
                    pass
                #training_data.append([partial_face, subdir])
        except NotADirectoryError as d:
            pass

training_data=list()
op='/Users/krishrana/AjnaLens/Resources/Datasets/RWMFD_part_2_partial'
predictor = dlib.shape_predictor('FaceDetection/shape_predictor_68_face_landmarks.dat')
get_training_data('/Users/krishrana/AjnaLens/Resources/Datasets/RWMFD_part_2_pro', op)