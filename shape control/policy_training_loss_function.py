import tensorflow as tf
import numpy as np
import cv2
from skimage.morphology import skeletonize
import scipy.io as sio
from contraction_loss_function import contraction_condition_loss

def policy_loss_function(control_policy,contraction_input_dim,initial_shape,target_shape_set,contraction_metric,policy_input_dim,robot_shape_model,action_dimension): 

    offline_data=sio.loadmat('offline_data.mat')
    random_select_action=offline_data.get('random_select_action')
    normalized_action=offline_data.get('normalized_action')
    action_dimension=11

    behaviour_cloning_loss=0.0
    for i in range(len(random_select_action)):
        image_name="image_training"+str(i+1)+".png"
        target_shape = cv2.imread(image_name)
        target_shape=target_shape/255.0
        target_shape=target_shape[:, :, :].sum(axis=2)
        target_shape[target_shape > 0] = 1.0

        image_name="image_training"+str(i)+".png"
        initial_state = cv2.imread(image_name)
        initial_state=initial_state/255.0
        initial_state=initial_state[:, :, :].sum(axis=2)
        initial_state[initial_state > 0] = 1.0

        model_input=np.dstack((target_shape,initial_state))
        model_input=model_input.reshape(1,model_input.shape[0],model_input.shape[1],model_input.shape[2])
        predicted_action=control_policy(model_input)

        selected_action=random_select_action[i]
        node_index=int(selected_action[:-1])
        sample_index=int(selected_action[-1])
        true_action=np.zeros((action_dimension,))
        true_action[node_index]=normalized_action[node_index,sample_index]

        behaviour_cloning_loss+=tf.reduce_sum(tf.square(true_action-predicted_action))



    contraction_loss=0.0
    convergence_rate=0.7
    convergence_step=3
    contraction_element_num=contraction_input_dim

    temp_shape=initial_shape
    temp_skeleton = skeletonize(temp_shape)
    temp_skeleton_line=np.array(list(zip(*np.where(temp_skeleton == True))))
    selected_contraction_element=np.linspace(0,len(temp_skeleton_line)-1,contraction_input_dim).astype(int)
    
    for k in range(len(target_shape_set)):
        target_shape=target_shape_set[k]
        target_skeleton = skeletonize(target_shape)
        target_skeleton_line=np.array(list(zip(*np.where(target_skeleton == True))))

        selected_target_state=target_skeleton_line[selected_contraction_element,:]
        selected_element_state=temp_skeleton_line[selected_contraction_element,:]      
        contraction_input=selected_element_state.reshape(1, contraction_element_num*2)
        fai_value=contraction_metric(contraction_input)
        fai_value=tf.reshape(fai_value,(contraction_element_num,contraction_element_num))
        contraction_matrix=tf.matmul(tf.transpose(fai_value), fai_value)+tf.eye(contraction_element_num)
        temp_state_error=tf.norm(selected_target_state-selected_element_state,axis=1)
        temp_state_error = tf.cast(temp_state_error, tf.float32) 
        temp_state_error=tf.reshape(temp_state_error,[1,contraction_element_num])
        distance_pre=tf.matmul(tf.matmul(temp_state_error,contraction_matrix), tf.transpose(temp_state_error))
    
        for _ in range(convergence_step):                    
            policy_input=np.dstack((target_shape,temp_shape))
            policy_input=policy_input.reshape(1,policy_input_dim[0],policy_input_dim[1],policy_input_dim[2])
            temp_action=control_policy(policy_input) 
            temp_action=temp_action.numpy()
            temp_action=np.tile(temp_action, int(target_shape.shape[1]/action_dimension)+1)
            temp_action=temp_action[:target_shape.shape[1]]
            temp_action=np.tile(temp_action,(target_shape.shape[0],1))
            model_input=np.dstack((initial_state,temp_action))
            model_input=model_input.reshape(1,model_input.shape[0],model_input.shape[1],model_input.shape[2])

            temp_shape=robot_shape_model(model_input)   
            temp_shape=temp_shape.numpy()  
            temp_shape=temp_shape.reshape(target_shape.shape)
            temp_skeleton = skeletonize(temp_shape)
            temp_skeleton_line=np.array(list(zip(*np.where(temp_skeleton == True))))

            selected_element_state=temp_skeleton_line[selected_contraction_element,:]
            contraction_input=selected_element_state.reshape(1, contraction_element_num*2)
            fai_value=contraction_metric(contraction_input)
            fai_value=tf.reshape(fai_value,(contraction_element_num,contraction_element_num))
            contraction_matrix=tf.matmul(tf.transpose(fai_value), fai_value)+tf.eye(contraction_element_num)
            temp_state_error=tf.norm(selected_target_state-selected_element_state,axis=1)
            temp_state_error = tf.cast(temp_state_error, tf.float32)
            temp_state_error=tf.reshape(temp_state_error,[1,contraction_element_num])
            distance_post=tf.matmul(tf.matmul(temp_state_error,contraction_matrix), tf.transpose(temp_state_error))

            contraction_distance_loss=tf.nn.relu(distance_post-convergence_rate*distance_pre)
            ccl=contraction_condition_loss(contraction_matrix)
            contraction_loss+=contraction_distance_loss+ccl
        
            distance_pre=distance_post


    total_loss=behaviour_cloning_loss+contraction_loss
    return total_loss



