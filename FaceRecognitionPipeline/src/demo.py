 # USAGE
# python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# import the necessary packages
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
import tensorflow.compat.v2 as tf
from sklearn.preprocessing import Normalizer
import dlib
from Tracking.centroidtracker import CentroidTracker
from Tracking.trackableobject import TrackableObject
from sklearn.preprocessing import Normalizer



VideoDirectoryPath="InputVideos/"
VideoOutDirectory='OutputVideos/'
faces_out_folder = "./CroppedFaces/"
# load our models from disk
print("[INFO] loading model...")
model_pkl="FaceDetection/weights.pkl"
tiny_faces_detector = TinyFacesDetector(model_pkl,use_gpu=True)
faceNet=load_model('recognition_models/DeepMaskFacev10.h5', compile=False)

#initial reference images
database=dict()
reference_names=[]
reference_path=os.path.join(faces_out_folder, 'Session1.4')
for img in os.listdir(reference_path):
    try:
        #img=str(img)+'.jpg'
        print(os.path.join(reference_path, img))
        reference_images=cv2.imread(os.path.join(reference_path, img))
        #initial_face_rects=tiny_faces_detector.detect(img_array,nms_thresh=0.1,prob_thresh=0.5,min_conf=0.9)
        #for bbox in initial_face_rects:
        #    reference_images = img_array[ int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]) ]
        reference_images=cv2.resize(reference_images, (112,112))
        reference_images=np.array(reference_images).reshape(-1,112,112,3)
        reference_images=reference_images/255
        embeds=faceNet.predict(reference_images)
        reference_names.append(str(img))
        database[str(img)]=embeds
    except Exception as e:
        pass
print(len(database))

#bb parameters
rect_threshold_area = 2200
font_style = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.4 

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
filename="4.webm"
vs = cv2.VideoCapture(VideoDirectoryPath+filename)
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
out = cv2.VideoWriter (str(VideoOutDirectory)+ str(filename)+'_Out_track.avi', cv2.VideoWriter_fourcc ('M', 'J', 'P', 'G'), 60, (frame_width,frame_height))

# initialize the list of object trackers and corresponding class
# labels
#trackers = []
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

	# if there are no object trackers we first need to detect objects
	# and then create a tracker for each object
	# Check for new faces after every 60 frames
	if total_frames % 60 == 0:
		print('Status: Detecting')
		num_frames=0
		trackers = []
		#face_list=[]
		face_region=[]

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

				# draw the bounding box
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)


	# otherwise, we've already performed detection so let's track
	# multiple objects
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
			try:
				face=cv2.resize(face, (112,112))
				face=np.array(face).reshape(-1,112,112,3)
				face=face/255
				face_list.append(face)
			except Exception as e:
				pass

		print('total_faces', len(face_list))
		objects = ct.update(rects)

		for (objectID, centroid), f, th in zip(objects.items(), face_list, face_region):
			distance=[]
		# check to see if a trackable object exists for the current
		# object ID
			to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)
				if total_frames>=60:
					embedding=faceNet.predict(f)
					for e in database:
						dist=np.linalg.norm(embedding-database[e])
						distance.append(dist)
					if min(distance)>1.2:
						name='Unknown ID:' + str(objectID)
						print('Name', str(objectID), 'Anchor', objectID)
						name_list.append([objectID, name])
					else:
						ind=distance.index(min(distance))
						name=str(reference_names[ind])
						print('Name', name, 'Anchor', objectID)
						name_list.append([objectID, name])


			
		# store the trackable object in our dictionary
			trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
			text = "ID {}".format(objectID)

			#The model takes inferences for the first 30 frames
			#to lock the identity of the person.
			if total_frames<30:
				embedding=faceNet.predict(f)
				for e in database:
					dist=np.linalg.norm(embedding-database[e])
					distance.append(dist)
				print('distance length', len(distance))
				if min(distance)>1.2:
					name='Unknown ID:' + str(objectID)
					print('Name', str(objectID), 'Anchor', objectID)
					name_list.append([objectID, name])
				else:
					ind=distance.index(min(distance))
					print('Index', ind)
					name=str(reference_names[ind])
					print('Name', name, 'Anchor', objectID)
					name_list.append([objectID, name])
			else:
				identity=[name_list[x][1] for x in range(len(name_list)) if name_list[x][0]==objectID]
				if len(identity)>0:
					name=max(set(identity), key=identity.count)
					name=str(name)
					print('Name', name, 'Anchor', objectID)
				else:
					name=''
			#name='Name: '+name
			#cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

			#if th>rect_threshold_area:
			cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
			cv2.putText(frame, name, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1)






			# draw the bounding box from the correlation object tracker
			

	# check to see if we should write the frame to disk
	if out is not None:
		out.write(frame)

	# show the output frame
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