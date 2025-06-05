import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio
from shape_class import Shape_model
from shape_model_training_loss_function import model_training_loss_function
import time
import cv2
import os
image_folder=r"D:/OneDrive/SMA_soft_arm/offline_data_20241014_163826"
os.chdir(image_folder)

#----------------------------------------------------------------------------------------
#train model 
offline_data=sio.loadmat('offline_data_20241014_163919.mat')
min_action_time=offline_data.get('each_node_min_action_time')
min_action_time=min_action_time[0,0]
max_action_time=offline_data.get('each_node_max_action_time')
max_action_time=max_action_time[0,0]

img = cv2.imread('image_training0.png')
model_output_dim=img.shape[0]*img.shape[1]
model_input_dim=np.array([img.shape[0],img.shape[1],2])

shape_model=Shape_model(input_dim=model_input_dim,output_dim=model_output_dim)
training_epochs=200
model_training_loss=np.zeros((training_epochs,1))
opt= tf.keras.optimizers.Adam(learning_rate=0.01)

temp=shape_model.model_param()
for i in range(training_epochs):
    print('training epoch: ', i+1)

    with tf.GradientTape() as t:
      t.watch(temp)
      model_loss = model_training_loss_function(shape_model,min_action_time,max_action_time)
    grads = t.gradient(model_loss, temp)
    opt.apply_gradients(zip(grads,temp))
     
    model_training_loss[i]=model_loss.numpy()
    print('training loss:',model_training_loss[i])


plt.figure(1)
plt.plot(model_training_loss)
plt.show() 


filename="robot_shape_model.h5"
shape_model.save_model(filename)









