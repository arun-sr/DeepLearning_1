import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #image of handwritten image 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0], cmap=plt.cm.binary)
#print(x_train[0])

#Normalize the values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0], cmap=plt.cm.binary) #binary to remove the color information
plt.show()
#print(x_train[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = x_train[0].shape))

# ReLU stands for rectified linear unit, and is a type of activation function. 
#Mathematically, it is defined as y = max(0, x).
#ReLUs are the most commonly used activation function in neural networks, especially in CNNs. 

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)


valuation_loss, valuation_accuracy = model.evaluate(x_test, y_test)
print(valuation_loss, valuation_accuracy)

model.save('epic_reader_num_model')

new_model = tf.keras.models.load_model('epic_reader_num_model')
predictions = new_model.predict([x_test])
print(predictions)

print("Predicted value is : ", np.argmax(predictions[1]))
print("Predicted value is : ", predictions[1])

plt.imshow(x_test[1], cmap=plt.cm.binary)
plt.show()
