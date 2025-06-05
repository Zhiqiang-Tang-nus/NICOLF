import tensorflow as tf
import numpy as np


##----------------------------------------------------------------------------------------
##estimated dynamic model loss
def model_param_update_loss_function(robot_shape_model, update_action, update_shape): 
    total_loss=0.0

    for i in range(len(update_action)):
        true_image = update_shape[i+1]
        initial_state = update_shape[i]
        action=update_action[i]

        model_input=np.dstack((initial_state,action))
        model_input=model_input.reshape(1,model_input.shape[0],model_input.shape[1],model_input.shape[2])
        predicted_image=robot_shape_model(model_input)
        predicted_image=tf.reshape(predicted_image,true_image.shape)

        total_loss+=tf.reduce_sum(tf.square(true_image-predicted_image))

    return total_loss



