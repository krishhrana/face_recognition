package org.tensorflow.lite.examples.detection.tflite;
import android.graphics.Bitmap;

import java.util.List;
import java.util.Objects;

public class Face {
    String name;
    String id;
    String face_embeddings;

    public Face (){

    }

    public Face(String name, String id, String face_embeddings) {
        this.name = name;
        this.id = id;
        this.face_embeddings=face_embeddings;
    }

    public String getName() {
        return name;
    }

    public String getId() {
        return id;
    }

    public String getFace_embeddings() { return face_embeddings; }
}
