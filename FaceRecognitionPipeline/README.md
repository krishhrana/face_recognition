# Face Recognition and Tracking 

## Requirements
``` Tensorflow: 2.1.0```
``` Tensorflow_addons: 0.9.1```
``` Opencv: 4.1.0```
``` Dlib: 19.18.0```
``` TinyFacesDetector module```

### Running inference on real-time videos
To run the inference on realtime videos, place the video in 'InputVideos' folder in the 'src' directory.

The module has to be run in two sessions 
1. Extract the faces form the video and save it for future inference.
2. Run real-time inference on videos.

In Session 1, run the 'save_new_faces.py' script on the Input video. The faces from the video are extracted and saved in 'CroppedFaces' folder. 

In Session 2, run 'demo.py' script. Set the reference folder to the folder containing the reference images and set the 'filename' to the video file name. The output video is saved in the 'OutputVideos' folder. Run the demo.py script.

``` python3 demo.py```

The reference embeddings are stored in a dictionary with {Name: embeddings} as key:value pairs. The indentity is assigned based on the minimum distance between the known face embeddings and the faces that appear in th video. Person is categorised as 'Unknown' if the euclidean distance is greater than the threshold which is set to '1.2'

``` filename = '10.webm' ```
``` python3 demo.py```

The TinyFacesDetector module for face detection is placed in the FaceDetection folder along with the weights.pkl file.

The face tracking system is based on dlib_correlation tracker coupled with centroid tracking.'Tracking' folder contains 'centroidtracker.py' and 'trackableobject.py' files . The 'centroidtracker.py' contains the code for centroid tracking. It used to generate and keep track of the ids assigned and remove the ids from memory after the faces have moved out of the frame for a certain period of time.


### Data and Backend processing
The 'face_landmarks.py' and 'PartialFacesGenerator.py' contains the scripts to visualise the face landamrks and generate and save the partial face images. These scripts have dependencies on the TinyFacesDetector module.

The BackendProcess.py contains a trial script to filter the noisy and duplicate images that may have been captured. It currently works with firebase and uses firebase-admin package. You need to have a service account to perform 'delete' operation. It is a trial script and needs further development. 

### Training Procedure
The 'training.ipynb' notebook contains the training procedure. It uses tensorflow's implementation of Triplet Loss function which is included in the tensorflow_addons. Check the version compaitbilty matrix of tfa to match with your version of tensorflow: https://github.com/tensorflow/addons
MobileNetv2 is used as the base_model and the top layers are added based on the MobileFaceNet paper. 
The image size is (112,112) and rescaling of 1./255 is applied.

You can run the umap_projection_generator notebook to generate vector embeddings on a test set and view it here: https://projector.tensorflow.org. 

