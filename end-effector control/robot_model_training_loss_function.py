import tensorflow as tf
import numpy as np
import scipy.io as sio

##----------------------------------------------------------------------------------------
##estimated dynamic model loss
def model_training_loss_function(robot_model,w_dim): 
    
    model_num=4
    reg_cof=1e-2
    state_min=np.asarray([-80, 310, -60])
    state_max=np.asarray([80, 370, 120])
    total_loss=0
    for i in range(model_num):
        if i==0:
            offline_data = sio.loadmat('0gram_offline_data.mat')
        if i==1:
            offline_data = sio.loadmat('10gram_tip_offline_data.mat')
        if i==2:
            offline_data = sio.loadmat('20gram_tip_offline_data.mat')
        # if i==3:
        #     offline_data = sio.loadmat('30gram_tip_offline_data.mat')

        offline_action_data=offline_data.get('action_record')
        offline_action_data=offline_action_data.astype('float32')
        offline_state_data=offline_data.get('state_record')
        offline_state_data=offline_state_data.astype('float32')
        offline_state_data=(state_max-offline_state_data)/(state_max-state_min)
        offline_num,state_dim=offline_state_data.shape
        input_train_data=np.concatenate((offline_state_data[:-1, :],
                                        offline_action_data),axis=1)
        output_train_data=offline_state_data[1:, :]

        G=robot_model(input_train_data)
        G=tf.nn.sigmoid(G)
        temp=tf.matmul(tf.transpose(G), G)+reg_cof*tf.eye(w_dim)
        temp=tf.linalg.inv(temp)
        temp=tf.matmul(temp, tf.transpose(G))        
        for j in range(state_dim):
            Y=output_train_data[:,j].reshape(offline_num-1,1)
            W=tf.matmul(temp, Y)
            model_estimated_output=tf.matmul(G,W)

            # model_estimated_output=state_max[j]-model_estimated_output*(state_max[j]-state_min[j])
            # Y=state_max[j]-Y*(state_max[j]-state_min[j])

            total_loss=total_loss+tf.reduce_sum(tf.square(model_estimated_output-Y)) 

    return total_loss



