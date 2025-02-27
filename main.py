from neuronalNetwork import trainModelAndSave
import numpy as np
from fastapi import FastAPI, File, UploadFile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/trainAndSaveModel")
def trainModel():
    result = trainModelAndSave()
    return result

@app.post("/analyze")
async def predictNumber(file: UploadFile = File(...)):
    # Leer la imagen subida
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    
    # Convertir la imagen a escala de grises y redimensionar
    image = image.convert('L')
    image = image.resize((28, 28))
    
    # Convertir la imagen a un array de numpy
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    
    plt.imshow(np.reshape(image_array, (28,28)))
    plt.show()
    
    mnist_model = load_model("modelo_mnist.h5")
    
    X_new_prep = image_array.reshape((1, 28*28))
    X_new_prep = X_new_prep.astype('float32') / 255
    
    # Hacer la predicción
    y_proba = mnist_model.predict(X_new_prep)
    
    y_pred = np.argmax(mnist_model.predict(X_new_prep), axis=-1)
    
    print(y_proba, 'd=====(￣▽￣*)b')

    return { "number": int(y_pred[0]) }