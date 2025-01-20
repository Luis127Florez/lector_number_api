from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

def trainModelAndSave():
    import matplotlib.pyplot as plt
    print(tf.__version__)

    mnist = datasets.mnist
    print(mnist)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape, '┗( T﹏T )┛')
    print(X_test.shape, '┗( T﹏T )┛')


    plt.figure(figsize=(20, 4))

    for index, digit in zip(range(1, 9), X_train[:8]):
        plt.subplot(1, 8, index)
        plt.imshow(np.reshape(digit, (28, 28)), cmap=plt.cm.gray)
        plt.title('Ejemplo: ' + str(index))
    plt.show()


    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    print(X_test.shape, 'X_test.shape')
    print(X_val.shape, 'X_val.shape')


    network = models.Sequential()

    network.add(layers.Dense(300, activation='relu', input_shape=(28*28,)))
    network.add(layers.Dense(100, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))

    network.summary()

    hidden1 = network.layers[1]

    weights, biases = hidden1.get_weights()

    print(weights, '（⊙ｏ⊙）')
    print(biases, '（⊙ｏ⊙）')

    network.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy', 'Precision']
    )

    X_train_prep = X_train.reshape((60000, 28*28))
    X_train_prep = X_train_prep.astype('float32') / 255

    X_test_prep = X_test.reshape((5000, 28*28))
    X_test_prep = X_test_prep.astype('float32') / 255

    X_val_prep = X_val.reshape((5000, 28*28))
    X_val_prep = X_val_prep.astype('float32') / 255

    from tensorflow.keras.utils import to_categorical
    y_train_prep = to_categorical(y_train)
    y_test_prep  = to_categorical(y_test)
    y_val_prep = to_categorical(y_val)

    print('train...')
    history = network.fit(
        X_train_prep,
        y_train_prep,
        epochs=30,
        validation_data=(X_val_prep, y_val_prep)
    )

    import pandas as pd
    import matplotlib.pyplot as plt

    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.2)
    plt.xlabel("epochs")
    plt.show()

    test_loss, test_acc, test_prec = network.evaluate(X_test_prep, y_test_prep)
    print(test_loss, '(๑•̀ㅂ•́)و✧')
    print(test_acc, '(๑•̀ㅂ•́)و✧')
    print(test_prec, '(๑•̀ㅂ•́)و✧')
    
    # Guardamos el modelo en disco
    network.save("modelo_mnist.h5")
    
    return {
        test_loss: test_loss,
        test_acc: test_acc,
        test_prec: test_prec,
    }
    
    
