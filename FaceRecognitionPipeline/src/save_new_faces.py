from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
from FaceDetection.TinyFacesDetector import TinyFacesDetector
import os
from Utils import Utils
from FaceAligner import FaceAligner
import glob
import util
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
import sklearn
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import dlib
from Tracking.centroidtracker import CentroidTracker
from Tracking.trackableobject import TrackableObject
from sklearn.preprocessing import Normalizer

VideoDirectoryPath="InputVideos/"
faces_out_folder = "./CroppedFaces/"
filename='7.webm'
output_folder= faces_out_folder+'Session1.'+filename[0]
Utils.mkdir_if_not_exist(output_folder)
# load our serialized model from disk
print("[INFO] loading model...")
model_pkl="FaceDetection/weights.pkl"
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)

rect_threshold_area = 2200
font_style = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.4 

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(VideoDirectoryPath+filename)
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

labels = []
name_list=[]

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
total_frames=0

while True:
	rects=[]
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break

	# frame from BGR to RGB ordering (dlib needs RGB ordering)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if total_frames % 60 == 0:
		print('Status: Detecting')
		num_frames=0
		trackers = []
		#face_list=[]
		face_region=[]

		# grab the frame dimensions and convert the frame to a blob
		face_rects=tiny_faces_detector.detect(frame,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
		# loop over the detections
		
		for bbox in face_rects:
			
			box = [ int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
			face=frame[ int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]) ]
			height, width, channels = face.shape
			face_size=height*width
			(startX, startY, endX, endY) = box
			
				# construct a dlib rectangle object from the bounding
				# box coordinates and start the correlation tracker

			t = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, endX, endY)
			t.start_track(rgb, rect)


				# update our set of trackers and corresponding class
				# labels
			#labels.append(id)
			trackers.append(t)

				# grab the corresponding class label for the detection
				# and draw the bounding box
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	else:
		face_list=[]
		print('Status Tracking')
		# loop over each of the trackers
		for t in trackers:
			#distance=[]
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			rects.append((startX, startY, endX, endY))
			face=frame[ startY:endY, startX:endX ]
			height, width, channels = face.shape
			face_area=height*width
			face_region.append(face_area)
			face_list.append(face)
			

		print('total_face', len(face_list))
		objects = ct.update(rects)

		for (objectID, centroid), f, th in zip(objects.items(), face_list, face_region):
			distance=[]
		# check to see if a trackable object exists for the current object ID
			to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)
				cv2.imwrite(output_folder+'/'+str(objectID)+'.jpg', f)
			trackableObjects[objectID] = to

			cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
			cv2.putText(frame, str(objectID), (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
	print('Frame: ',total_frames)
	total_frames=total_frames+1
	num_frames=num_frames+1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()











