import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_class import Robot_model
from robot_model_training_loss_function import  model_training_loss_function

#----------------------------------------------------------------------------------------
#train model 
state_dimension=3
action_dimension=6
w_dim=4
robot_model=Robot_model(dim_state_control=state_dimension+action_dimension, dim_W=w_dim)
training_epochs=100
model_training_loss=np.zeros((training_epochs,1))
model_test_loss=np.zeros((training_epochs,1))
opt= tf.keras.optimizers.Adam(learning_rate=0.01)

temp=robot_model.model_param()
for i in range(training_epochs):
    print('training epoch: ', i+1)

    with tf.GradientTape() as t:
      t.watch(temp)
      model_loss = model_training_loss_function(robot_model,w_dim)
    grads = t.gradient(model_loss, temp)
    opt.apply_gradients(zip(grads,temp))

     
    model_training_loss[i]=model_loss.numpy()
    print('training loss:',model_training_loss[i])



filename="robot_model.h5"
robot_model.save_model(filename)

plt.figure(1)
plt.plot(model_training_loss)
plt.show() 








