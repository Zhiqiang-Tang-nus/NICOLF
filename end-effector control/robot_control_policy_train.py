import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_class import Control_policy, Contration_metric    
from control_policy_training_loss_function import  control_policy_loss_function
from generate_test_targets_xyz import generate_training_target
import warnings
warnings.filterwarnings("ignore")



##----------------------------------------------------------------------------------------
##train contration metric and controller
state_dimension=3
action_dimension=6
contration_metric=Contration_metric(len_states=state_dimension)
control_policy=Control_policy(dim_target_state=2*state_dimension,dim_output=action_dimension)
G_model=tf.keras.models.load_model('robot_model_meta_20231107_123729.h5')

num=10
reference_states_cell=[]
initial_states_cell=[]
for i in range(num):
  reference_states,initial_states=generate_training_target()
  reference_states_cell.append(reference_states)
  initial_states_cell.append(initial_states) 
reference_states_cell=np.asarray(reference_states_cell)
initial_states_cell=np.asarray(initial_states_cell)

training_epochs=100
loss_record=np.zeros((training_epochs,1)).astype('float32')
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
temp=control_policy.model_param()+contration_metric.model_param()
for k in range(training_epochs):
    print('training epoch', k+1)

    with tf.GradientTape() as t:
      t.watch(temp)
      training_loss = control_policy_loss_function(control_policy,contration_metric,G_model,reference_states_cell,initial_states_cell)
    grads = t.gradient(training_loss, temp)
    opt.apply_gradients(zip(grads,temp))

    loss_record[k] = training_loss.numpy()         
    print('training loss:',loss_record[k])

plt.plot(loss_record)

filename="control_policy.h5"
control_policy.save_model(filename)
filename="contration_metric.h5"
contration_metric.save_model(filename)









