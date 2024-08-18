import pandas as pd
import tensorflow as tf

# Importing the dataset
train_set, valid_set, test_set = pd.read_pickle("data/mnist.pkl.gz")

# Splitting the dataset into the Training set, Validation set and Test set on X and y bases
X_train, y_train = train_set
X_valid, y_valid = valid_set
X_test, y_test = test_set

# Reshaping the dataset
X_train = X_train.reshape(-1, 28, 28, 1)
X_valid = X_valid.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Convolution 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

# Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the validation set
cnn.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=10)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")

"""
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9895 - loss: 0.0489   
Test accuracy: 0.9916999936103821
Test loss: 0.041889917105436325
"""
