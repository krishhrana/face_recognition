import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import cv2
import urllib
import numpy as np
import dlib
from FaceDetection.TinyFacesDetector import TinyFacesDetector
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# Credential and configuration of firebase storage and realtime database
cred=credentials.Certificate('your-credential-file.json')
firebase_admin.initialize_app(cred, {
    "databaseURL":"database-url",
    "storageBucket": "storage-bucket-url",
    "projectId": "project-id",
    "serviceAccount": "your-service-account-file.json"
})


global graph
graph = tf.compat.v1.get_default_graph()
ref=db.reference('Face Data')
bucket = storage.bucket()
tiny_faces_detector=TinyFacesDetector("FaceDetection/weights.pkl",use_gpu=True)

def isBlurry(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur=cv2.Laplacian(img, cv2.CV_64F).var()
    if blur<100:
        return True, blur
    else:
        return False, blur

def cleanDatabase(ids):
    imageList=[]
    names=[]
    for i in ids:
        filename=i+'.jpg'
        blob = bucket.blob("faces/"+filename)
        urls=blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        #print(urls)
        resp = urllib.request.urlopen(urls)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        check, blur=isBlurry(image)
        if check is False:
            with graph.as_default():
                rects=tiny_faces_detector.detect(image,nms_thresh=0.1,prob_thresh=0.9,min_conf=0.9)
            if len(rects)!=0: 
                image=cv2.resize(image, (112,112))
                image=image/255
                image=np.array(image).reshape(-1,112,112,3)
                imageList.append(image)
                names.append(filename)
            else:
                blob = bucket.blob('faces/'+filename)
                blob.delete()
                ref.child(i).delete()
        else:
            blob = bucket.blob('faces/'+filename)
            blob.delete()
            ref.child(i).delete()
    return imageList, names


def getEmbeddings(imageList):
    model=load_model('recognition_models/DeepMaskFacev10.h5', compile=False)
    pred=model.predict(imageList)
    return pred


def deleteSimilarFaces(prediction_list, reference_list, name_list):
    cleanup_list=reference_list
    similar_faces=[]
    for i in range(len(prediction_list)):
        for j in range(len(reference_list)):
            dist= np.linalg.norm(prediction_list[i]-reference_list[j])
            print(dist)
            if dist!=0.0:
                if dist<0.685:
                    print(name_list[i], name_list[j])
                    blob = bucket.blob('faces/'+name_list[i])
                    blob.delete()
                    child_node=name_list[i].replace('.jpg', '')
                    ref.child(child_node).delete()
                    print(name_list[i]+'DELETED!')
                    del name_list[i]
                    cleanup_list = np.delete(cleanup_list, i)
    print(len(name_list), len(cleanup_list))
    return name_list, cleanup_list


#listen function for realtime updates and changes in database.
new_embeds_list=[]
clean_names=[]
def listen(event):
    id_list=[]
    new_id=[]
    new_id_list=[]
    print(event.path)
    if event.path=='/':
        for i in event.data:
            print('ID:'+ i)
            id_list.append(i)
        faces, names=cleanDatabase(id_list)
        faces=np.array(faces).reshape(-1,112,112,3)
        print(faces.shape)
        embeddings=getEmbeddings(faces)
        clean_names, new_embeds_list=deleteSimilarFaces(embeddings, embeddings, names)
        
        
    else:
        for i in event.data:
            new_id_list.append(event.data[i])
        
        new_faces, new_names=cleanDatabase(new_id_list)
        new_embeddings=getEmbeddings(new_faces)
        new_clean_names, new_cleanup_list=deleteSimilarFaces(new_embeddings, 
                                                             new_embeds_list, new_names)
        
        np.concatenate(new_embeds_list, new_cleanup_list)
    
#Starts the eventListner
listner=ref.listen(listen)
