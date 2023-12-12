# MNIST Classification using CNN

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import SVG
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
import os

np.random.seed(100)

# def show_history(acc, loss):
#     plt.plot(acc)
#     plt.plot(loss)
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
#     plt.show()

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to match the Conv2D input shape
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# Retrieve a sample for reference
m = x_train[0]

# Convert input to float for normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize pixel values to be between 0 and 1
x_train /= 255
x_test /= 255

# Convert labels to one-hot vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Check if a pre-existing model file exists
if os.path.exists('mnist_cnn.h5'):
    print('Model existed! Load model from file')
    model = load_model('mnist_cnn.h5')
else:
    print('Train new model')

    # Build a Sequential model
    model = Sequential()

    # Add Convolutional and Pooling layers
    model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # Save a visualization of the model architecture
    plot_model(model, to_file='model.png',show_shapes=True)

    # Compile the model with categorical crossentropy loss and SGD optimizer
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])
    
    # Train the model and save it
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2, validation_data=(x_train, y_train))
    model.save('mnist_cnn.h5')
    
    # Save accuracy and loss data for later visualization
    acc_data = np.array(history.history['accuracy'])
    print("Accuracy Data \n:", acc_data)
    loss_data = np.array(history.history['loss'])
    print("Loss Data \n:", loss_data)

    np.save('data1.npy', acc_data)
    np.save('data2.npy', loss_data)

# Load saved accuracy and loss data for visualization
# x = np.load('data1.npy')
# y = np.load('data2.npy')

# Show the training history
# show_history(x,y)

print('Evaluating model')

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=1)

# Make predictions on the test set
prediction = model.predict(x_test)

# Convert predictions to class labels
prediction = np.argmax(prediction, axis=1)
y_test = np.argmax(y_test,axis=1)

# Compute confusion matrix and performance metrics
matrix = confusion_matrix(y_test, prediction)
f1score = f1_score(y_test, prediction, average='weighted')
precision = precision_score(y_test, prediction,average='weighted')

# Display evaluation results
print('Test score {0}'.format(score))
print('F1 score {0}'.format(f1score))
print('Precision score {0}'.format(precision))
print('Confusion matrix:\n{0}'.format(matrix))


