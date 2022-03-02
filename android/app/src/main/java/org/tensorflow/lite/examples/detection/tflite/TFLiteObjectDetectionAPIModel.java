/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Trace;
import android.util.Log;
import android.util.Pair;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

import androidx.annotation.NonNull;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel
        implements SimilarityClassifier {

  private static final Logger LOGGER = new Logger();

  //private static final int OUTPUT_SIZE = 512;
  private static final int OUTPUT_SIZE = 128;
  private static final int OUTPUT_HEIGHT = 320;
  private  static final int OUTPUT_WIDTH = 640;
  private static final int OUTPUT_CHANNELS = 1;

  // Only return this many results.
  private static final int NUM_DETECTIONS = 1;

  // Float model
  //private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 255.0f;

  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][][][] embeedings;

  private float[][] database_embeddings;

  private String final_database_embeddings;

  private ByteBuffer imgData;

  private Interpreter tfLite;
  public Bitmap image;
  private Bitmap faceBmp = null;
  private ByteBuffer initial_imgData;

// Face Mask Detector Output
  private float[][] output;


// Registering to the database
  DatabaseReference databaseFace;
  private HashMap<String, Recognition> registered = new HashMap<>();
  public void database_register(String name, Recognition rec){
      databaseFace=FirebaseDatabase.getInstance().getReference();
      String id =databaseFace.push().getKey();
      //Storing the Face metadata
      Face face =new Face();
      face.name=name;
      face.id=rec.getTracking_id().toString();
      database_embeddings= (float[][]) rec.getExtra();
      final_database_embeddings= Arrays.deepToString(database_embeddings);
      face.face_embeddings=final_database_embeddings;
      databaseFace.child("Face Data/"+face.id).setValue(face);
      databaseFace.child("Total people scanned").setValue(rec.getTracking_id()+1);

      //Storing the face image
      if (rec.getCrop() != null){
          image=rec.getCrop();
          FirebaseStorage storage = FirebaseStorage.getInstance();
          StorageReference storageRef = storage.getReferenceFromUrl("gs://trial-a6e1b.appspot.com");
          StorageReference mountainImagesRef = storageRef.child("faces/" + face.id + ".jpg");

          ByteArrayOutputStream baos = new ByteArrayOutputStream();
          image.compress(Bitmap.CompressFormat.JPEG, 20, baos);
          byte[] data = baos.toByteArray();
          UploadTask uploadTask = mountainImagesRef.putBytes(data);

          uploadTask.addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception exception) {
                  // Handle unsuccessful uploads
              }
          }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
              @Override
              public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                  // taskSnapshot.getMetadata() contains file metadata such as size, content-type, and download URL.
                  Log.d("downloadUrl-->", "");
              }
          });
      }
  }


// Internal Reference
  public void register(String name, Recognition rec) { registered.put(name, rec); }


  //Retrieval from the database (name,  Face embeddings)
  HashMap<String,Object> database_registered = new HashMap<>();
  public  HashMap<String, Object> databaseRetrieve(){
      DatabaseReference databaseFace;
      databaseFace=FirebaseDatabase.getInstance().getReference().child("Face Data");

      databaseFace.addValueEventListener(new ValueEventListener() {
          @Override
          public void onDataChange(@NonNull DataSnapshot snapshot) {
              for (DataSnapshot snapshot1: snapshot.getChildren()){
                  String name=snapshot1.child("name").getValue().toString();
                  String initial_embedding= (String) snapshot1.child("face_embeddings").getValue();
                  initial_embedding=initial_embedding.replace("[","");
                  initial_embedding=initial_embedding.replace("]", "");
                  String[] parts = initial_embedding.split(", ");
                  float[] final_embeddings=new float[parts.length];
                  for (int i=0; i<parts.length;i++){
                      float j= Float.parseFloat(parts[i]);
                      final_embeddings[i]=j;
                  }
                  database_registered.put(name, final_embeddings);
                  LOGGER.i("RETRIEVED"+final_embeddings.toString());
              }
          }
          @Override
          public void onCancelled(@NonNull DatabaseError error) {
          }
      });

      LOGGER.i("DATABASE LENGTH"+database_registered.size());

      return database_registered;
  }


  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static SimilarityClassifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }


  // looks for the nearest embeeding in the dataset (using L2 norm)
  // and retrurns the pair <id, distance>
  private Pair<String, Float> findNearest(float[] emb) {
    Pair<String, Float> ret = null;
    for (Map.Entry<String, Object> entry : database_registered.entrySet()) {
        final String name = entry.getKey();
        final float[] knownEmb = (float[]) entry.getValue();

        float distance = 0;
        for (int i = 0; i < emb.length; i++) {
              float diff = emb[i] - knownEmb[i];
              distance += diff*diff;
        }
        distance = (float) Math.sqrt(distance);
        if (ret == null || distance < ret.second) {
            ret = new Pair<>(name, distance);
        }
    }
    return ret;
  }

  private float[][][][] applyColor(float[][][][]disp){

      Co

  }



  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra, String maskPredictor) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF)) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF)) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF)) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");


    Object[] inputArray = {imgData};

    Trace.endSection();
// Here outputMap is changed to fit the Face Mask detector
      float distance = Float.MAX_VALUE;
      String id = "0";
      String label = "?";
      String id_mask="?";
      String label_mask="?";
      float confidence_mask=0;

    if (maskPredictor.equals("Face")){
        Map<Integer, Object> outputMap = new HashMap<>();
        embeedings = new float[1][OUTPUT_HEIGHT] [OUTPUT_WIDTH] [OUTPUT_CHANNELS];
        outputMap.put(0, embeedings);


        // Run the inference call.
        Trace.beginSection("run");
        //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        if (database_registered.size() > 0) {
            //LOGGER.i("dataset SIZE: " + registered.size());
            final Pair<String, Float> nearest = findNearest(embeedings[0]);
            if (nearest != null) {

                final String name = nearest.first;
                label = name;
                distance = nearest.second;
                LOGGER.i("nearest: " + name + " - distance: " + distance);
                 //!!!!!!!!UPDATE!!!!!!!!!!!!
                 if(distance<=1.2f){
                    storeExtra=false;
                }
            }
        }
    }

    Trace.endSection();
    return colour_depth;
  }



  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
