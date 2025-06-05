import tensorflow as tf
import numpy as np

def policy_param_update_loss_function(policy_function,policy_input_dim,target_shape,robot_shape_model,initial_shape): 

    policy_input=np.dstack((target_shape,initial_shape))
    policy_input=policy_input.reshape(1,policy_input_dim[0],policy_input_dim[1],policy_input_dim[2])
    policy_action=policy_function(policy_input) 

    model_input=np.dstack((initial_shape,policy_action))
    model_input=model_input.reshape(1,model_input.shape[0],model_input.shape[1],model_input.shape[2])

    temp_shape=robot_shape_model(model_input)  
    temp_shape=tf.reshape(temp_shape, shape=target_shape.shape) 

    loss=tf.reduce_sum(tf.square(target_shape-temp_shape))

    return loss



