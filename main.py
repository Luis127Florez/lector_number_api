from neuronalNetwork import trainModelAndSave
from typing import Union
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/trainAndSaveModel")
def trainModel():
    result = trainModelAndSave()
    return result

def array_from_bytes(file: bytes, extension):
    image = io.BytesIO(file)
    with image:
        image = mpimg.imread(image, format=extension)
    return image

def process_image(image):
    tsImage = image[:, :, :3]
    tsImage = tsImage / 255
    tsImage = 1 - tsImage
    tsImage = tf.image.rgb_to_grayscale(tsImage, name=None)
    tsImage = tf.image.resize(tsImage, (28, 28))
    tsImage = np.expand_dims(tsImage, 0)
    return tsImage

@app.post("/analyze")
async def predictNumber (file: UploadFile = File(...)):
    # Lee el archivo subido
    
    extension = file.content_type.split('/')[1]

    # Get file as an array from bytes
    image = array_from_bytes(file.file.read(), extension)
    
    image = process_image(image)
    
    print(image)
    
    plt.imshow(np.reshape(image, (28,28)))
    plt.show()
    
    print('image')
    print('image')
    print('image')
    
    X_new_prep = image.reshape((1, 28*28))
    X_new_prep = X_new_prep.astype('float32') / 255.0
    
    # Cargamos el modelo de disco
    mnist_model = load_model("modelo_mnist.h5")
    
    # Realizamos una nueva prediccion
    y_pred = np.argmax(mnist_model.predict(image), axis=-1)
    
    print(y_pred, 'd=====(￣▽￣*)b')

    return { "number": int(y_pred[0]) }