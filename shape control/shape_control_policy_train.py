import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio
from shape_class import Shape_control_policy,Contraction_metric
from policy_training_loss_function import policy_loss_function
import cv2
import os
image_folder=r"D:/OneDrive/SMA_soft_arm/offline_data_20241014_161812"
os.chdir(image_folder)


#----------------------------------------------------------------------------------------
#train model 
initial_shape = cv2.imread('image_training0.png')
policy_input_dim=np.array([initial_shape.shape[0],initial_shape.shape[1],2])
action_dimension=11
control_policy=Shape_control_policy(input_dim=policy_input_dim,output_dim=action_dimension)

contraction_input_dim=10
contraction_output_dim=contraction_input_dim*contraction_input_dim
contraction_metric=Contraction_metric(input_dim=contraction_input_dim*2,output_dim=contraction_output_dim)

robot_shape_model=tf.keras.models.load_model('robot_shape_model.h5')

training_epochs=200
policy_training_loss=np.zeros((training_epochs,1))
opt= tf.keras.optimizers.Adam(learning_rate=0.01)

temp=control_policy.model_param()+contraction_metric.model_param()
for i in range(training_epochs):
    print('training epoch: ', i+1)

    with tf.GradientTape() as t:
      t.watch(temp)
      policy_loss = policy_loss_function(control_policy,contraction_input_dim,initial_shape,
                                         target_shape_set,contraction_metric,policy_input_dim,
                                         robot_shape_model,action_dimension)
    grads = t.gradient(policy_loss, temp)
    opt.apply_gradients(zip(grads,temp))

     
    policy_training_loss[i]=policy_loss.numpy()
    print('training loss:',policy_training_loss[i])


filename="shape_control_policy.h5"
control_policy.save_model(filename)

plt.figure(1)
plt.plot(policy_training_loss)
plt.show() 









