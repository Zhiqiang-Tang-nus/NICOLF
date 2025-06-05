import numpy as np
import tensorflow as tf

def model_param_update_function(G_model,input_train_data,output_train_data):

    G=G_model(input_train_data)
    G=tf.nn.sigmoid(G)
    w_dim=G.shape[1]
    data_num,state_dimension=output_train_data.shape
    model_param=np.zeros((w_dim,state_dimension))
    temp=np.matmul(np.transpose(G), G)+0.01*np.eye(w_dim)
    temp=np.linalg.inv(temp)
    temp=np.matmul(temp, np.transpose(G))       
    for j in range(state_dimension):
        Y=output_train_data[:,j].reshape(data_num,1)
        model_param[:,j]=np.matmul(temp, Y).reshape(w_dim,)   

    return model_param


