import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import train
import cv2 as cv

mnist=input_data.read_data_sets("E:/python_study/maching learning/Project/mnistagain/MNIST_data/",one_hot=True)

'''
img=cv.imread("2_1.png")
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(img.shape)
img=img.reshape(-1,28,28,1)
print(img.shape)
'''

index=122
img=np.reshape(mnist.test.images[index],[28,28])
plt.imshow(img,cmap=plt.cm.gray)
plt.show()
img=np.reshape(mnist.test.images[index],[1,28,28,1])

y_output=train.inference(tf.cast(img,tf.float32),False,None)
prediction = tf.argmax(input=y_output, axis=1)


variable_average=tf.train.ExponentialMovingAverage(decay=0.99)
saver=tf.train.Saver(variable_average.variables_to_restore())



with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    saver.restore(sess, "E:/python_study/maching learning/Project/zerotoone/mnist-gpu/model/mnist_model-29001")

    print(sess.run(prediction))


