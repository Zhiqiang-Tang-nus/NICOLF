import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tensorflow as tf
import serial
import pyrealsense2 as realsense
import scipy.io as sio
from exp_generate_test_targets import generate_test_target
from online_policy_update_function import policy_param_update_function
from online_model_update_function import model_param_update_function
import time
import warnings
warnings.filterwarnings("ignore")
from realsense_camera_sensor import POI_position
from motor_control_box import motor_control


# # -------------------------------------------------------------------------------------------------------
# # control box setup
dev = serial.Serial('COM4', baudrate=115200)
state_dimension=3
action_dimension=6
base=370*np.ones((1,action_dimension))
scale=np.asarray([50, 60, 50, 70, 60, 60]).reshape(1,action_dimension)
command=np.asarray([420, 430, 420, 440, 430, 430])
DATA_to_write = motor_control(command)
dev.write(bytearray(DATA_to_write))
time.sleep(1)


# # -------------------------------------------------------------------------------------------------------
# # camera setup
pipe = realsense.pipeline()
pipe.start()
# give camera Auto-Exposure time to adjust
for x in range(100):
  pipe.wait_for_frames()
align_to = realsense.stream.color     
alignedFs = realsense.align(align_to)
pointcloud = realsense.pointcloud()


# # -------------------------------------------------------------------------------------------------------
# # algorithm setup
control_policy=tf.keras.models.load_model('control_policy.h5')
G_model=tf.keras.models.load_model('robot_model.h5')
model_param=np.loadtxt('W_0gram_tip.dat')

reference_states=generate_test_target()
session_period=len(reference_states)
reference_states=np.tile(reference_states, (6, 1))
total_period=len(reference_states)
test_step=reference_states.shape[0]
state_min=np.asarray([-80, 310, -60])
state_max=np.asarray([80, 370, 120])
state_range=state_max-state_min
state_record=np.zeros((test_step+1,state_dimension))
action_record=np.zeros((test_step,action_dimension))
command_record=np.zeros((test_step,action_dimension))
tracking_error=np.zeros((test_step,1))
modeling_error=np.zeros((test_step,1))
model_state_record=np.zeros((test_step+1,state_dimension))
policy_param=np.ones((1,action_dimension))



# # -------------------------------------------------------------------------------------------------------
# # exp start
current_state=POI_position(pipe, alignedFs, pointcloud)
current_state=(state_max-state_record[0,:])/state_range
state_record[0,:]=current_state
target_state=reference_states[0,:]
for i in range(0,test_step):
  # # compute action  
  input_to_control_policy=np.concatenate((current_state,target_state),axis=1)
  control_signal=tf.nn.sigmoid(control_policy(input_to_control_policy)).numpy()
  action_record[i,:]=np.multiply(policy_param, control_signal)
  
  # # action=1: pull most; action=0: no pulling
  command_record[i,:]=np.multiply(1-action_record[i,:],scale)+base
  DATA_to_write = motor_control(command_record[i,:])
  dev.write(bytearray(DATA_to_write))
  
  current_state=POI_position(pipe, alignedFs, pointcloud)
  current_state=(state_max-state_record[i+1,:])/state_range
  state_record[i+1,:]=current_state

  # # update model parameters
  input_train_data=np.concatenate((state_record[0:i, :],action_record[0:i, :]),axis=1)
  output_train_data=state_record[1:i+1, :]
  model_param=model_param_update_function(G_model,input_train_data,output_train_data)

  # # update policy parameters
  policy_param=policy_param_update_function(control_signal, G_model, model_param,current_state,target_state)


# # -------------------------------------------------------------------------------------------------------
# # exp end
command=np.asarray([420, 430, 420, 440, 430, 430])
DATA_to_write = motor_control(command)
dev.write(bytearray(DATA_to_write))
time.sleep(1)


dev.close()
pipe.stop()

mdic = {"state_record": state_record, 
        "reference_states": reference_states, 
        "action_record": action_record}
exp_data="exp_data_"+time.strftime("%Y%m%d_%H%M%S")+".mat"
sio.savemat(exp_data, mdic)



