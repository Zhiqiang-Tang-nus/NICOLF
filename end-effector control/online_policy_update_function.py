import numpy as np
import tensorflow as tf
from numpy import linalg as LA

def policy_param_update_function(control_signal, G_model, Wm, current_state, target_state):

    policy_param_dim=control_signal.shape[1]
    policy_param=np.ones((1,policy_param_dim))
    query_number=10
    batch_size=5
    sample_rate=1
    sample_mean=0
    sample_var=0.1
    for i in range(query_number):
        loss_collect=np.ones((batch_size+1,1))
        policy_param_collect=[]

        for j in range(batch_size+1):  
            sample_step=(1-j/batch_size)*sample_rate
            sample_policy_param=policy_param+sample_step*np.random.normal(sample_mean,sample_var,size=(1,policy_param_dim))
            sample_policy_param=np.clip(sample_policy_param,0,1)

            control_action=np.multiply(sample_policy_param, control_signal)
            test_input=np.concatenate((current_state,control_action),axis=1)
            G=tf.nn.sigmoid(G_model(test_input))
            sapmle_state=np.matmul(G,Wm)

            loss_collect[j]=LA.norm(sapmle_state-target_state)
            policy_param_collect.append(sample_policy_param)


        min_index=np.argmin(loss_collect)
        policy_param=policy_param_collect[min_index]


    return policy_param


