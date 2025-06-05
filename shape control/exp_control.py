import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import serial
import time
import cv2
import scipy.io as sio
import tensorflow as tf
from numpy import linalg as LA
from online_image_processing import image_processing
from target_image_processing import target_processing
from online_policy_update_function import policy_param_update_loss_function
from model_update_loss_function import model_param_update_loss_function


# set target shape
target_shape = cv2.imread("drawn_target_shape.png")
target_shape=target_processing(target_shape)

# # arduino setup
dev_arm = serial.Serial('COM6', baudrate=9600)


# # camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)
# give camera Auto-Exposure time to adjust
for x in range(100):
  pipeline.wait_for_frames()


# load model and policy
robot_shape_model = tf.keras.models.load_model('robot_shape_model.h5')
original_shape_model = tf.keras.models.load_model('robot_shape_model.h5')
shape_control_policy=tf.keras.models.load_model('shape_control_policy.h5')
original_shape_control_policy=tf.keras.models.load_model('shape_control_policy.h5')
for layer in robot_shape_model.layers[:-1]:
    layer.trainable = False
for layer in original_shape_model.layers[:-1]:
    layer.trainable = False
for layer in shape_control_policy.layers[:-1]:
    layer.trainable = False
for layer in original_shape_control_policy.layers[:-1]:
    layer.trainable = False

opt_model=tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
opt_policy=tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
original_model_param=original_shape_model.trainable_variables
original_policy_param=original_shape_control_policy.trainable_variables


# # capture initial state
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
image_name="robot_shape"+str(0)+".png"
cv2.imwrite(image_name, color_image)
robot_shape=image_processing(color_image)
policy_input_dim=np.array([robot_shape.shape[0],robot_shape.shape[1],2])

# record variable
action_dimension=11
update_data_len=3
model_gradient_step=3
policy_gradient_step=3
time_step=20
robot_shape_record=[]
shape_error_record=[]
robot_action_record=[]
update_action=[]
update_shape=[]
node_record=[]

shape_error=LA.norm(target_shape-robot_shape)
shape_error_record.append(shape_error)
update_shape.append(robot_shape)
robot_shape_record.append(robot_shape)
for i in range(time_step):
    # # compute action
    policy_input=np.dstack((target_shape,robot_shape))
    policy_input=policy_input.reshape(1,policy_input.shape[0],policy_input.shape[1],policy_input.shape[2])
    action=shape_control_policy(policy_input) 
    action=action.numpy().reshape(action_dimension,)
    dev_arm.write(action.encode())

    # # capture robot state
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    image_name="robot_shape"+str(i+1)+".png"
    cv2.imwrite(image_name, color_image)
    robot_shape=image_processing(color_image)

    # # termination condition
    shape_error=LA.norm(target_shape-robot_shape)
    if shape_error>shape_error_record[-1]:
        break
    robot_shape_record.append(robot_shape)
    shape_error_record.append(shape_error)
    robot_action_record.append(action)


# # update model parameters
    update_action.append(action)
    update_shape.append(robot_shape)
    if len(update_action)>update_data_len:
        update_action=update_action[(len(update_action)-update_data_len):]
        update_shape=update_shape[(len(update_shape)-update_data_len-1):]

    robot_shape_model.layers[-1].set_weights(original_model_param)
    temp=robot_shape_model.trainable_variables
    for _ in range(model_gradient_step):
        with tf.GradientTape() as t:
            t.watch(temp)
            current_loss = model_param_update_loss_function(robot_shape_model, update_action, update_shape)
        grads = t.gradient(current_loss, temp)
        opt_model.apply_gradients(zip(grads,temp))


# # update policy parameters
    shape_control_policy.layers[-1].set_weights(original_policy_param)
    temp=shape_control_policy.trainable_variables
    for _ in range(policy_gradient_step):
        with tf.GradientTape() as t:
            t.watch(temp)
            current_loss = policy_param_update_loss_function(shape_control_policy,policy_input_dim,target_shape,robot_shape_model,robot_shape)
        grads = t.gradient(current_loss, temp)
        # print(grads)
        opt_policy.apply_gradients(zip(grads,temp))


mdic = {"robot_shape_record": robot_shape_record,
        "robot_action_record":robot_action_record,}
exp_data="exp_data_"+time.strftime("%Y%m%d_%H%M%S")+".mat"
sio.savemat(exp_data, mdic)

pipeline.stop()
dev_arm.close()

print('end')

