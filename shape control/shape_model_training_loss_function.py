import tensorflow as tf
import numpy as np
import cv2
import os
import scipy.io as sio


def model_training_loss_function(image_model,min_action_time,max_action_time): 
    total_loss=0

    image_folder=r"D:/OneDrive/SMA_soft_arm/offline_data_20241014_163826"
    os.chdir(image_folder)
    offline_data=sio.loadmat('offline_data_20241014_163919.mat')

    random_select_action=offline_data.get('random_select_action')
    heating_time=offline_data.get('heating_time')
    normalized_action=(heating_time-min_action_time)/(max_action_time-min_action_time)
    action_dimension=heating_time.shape[0]
    for i in range(len(random_select_action)):
        image_name="image_training"+str(i+1)+".png"
        true_image = cv2.imread(image_name)
        true_image=true_image/255.0
        true_image=true_image[:, :, :].sum(axis=2)
        true_image[true_image > 0] = 1.0

        image_name="image_training"+str(i)+".png"
        initial_state = cv2.imread(image_name)
        initial_state=initial_state/255.0
        initial_state=initial_state[:, :, :].sum(axis=2)
        initial_state[initial_state > 0] = 1.0

        selected_action=random_select_action[i]
        node_index=int(selected_action[:-1])
        sample_index=int(selected_action[-1])
        action=np.zeros((action_dimension,))
        action[node_index]=normalized_action[node_index,sample_index]
        action=np.tile(action, int(true_image.shape[1]/action_dimension)+1)
        action=action[:true_image.shape[1]]
        action=np.tile(action,(true_image.shape[0],1))

        model_input=np.dstack((initial_state,action))
        model_input=model_input.reshape(1,model_input.shape[0],model_input.shape[1],model_input.shape[2])


        predicted_image=image_model(model_input)
        predicted_image=tf.reshape(predicted_image,true_image.shape)

        mask_loss=tf.reduce_sum(tf.square(true_image-predicted_image))

        total_loss=total_loss+mask_loss



    return total_loss



