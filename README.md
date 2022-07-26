# Face Recognition

## Description:
#### 1. The known faces stored in the directory are loaded in the memory and their face embeddings are generated using the deep learning model and stored in a dictionary with {Name: embeddings} as key- value pairs.
2. The input video is loaded and the face detector (TinyFacesDetector) detects the faces in the frame and passes the face rects to the Dlib correlation tracker to track the faces.
3. For the first 30 frames of the video, the face embeddings are calculated and measured against the known faces to assign identity. The identity of a person is locked based on a voting system where the identity which is assigned the most number of times in the initial 30 frames is chosen.
4. After every 60 frames, the face detector again checks if any new faces have entered the frame and assign tracking id and identity to them.
