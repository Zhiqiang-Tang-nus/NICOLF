import tensorflow as tf
import numpy as np
from contraction_loss_function import contraction_condition_loss


def control_policy_loss_function(control_policy,contration_metric,G_model,reference_states_cell,initial_states_cell):
  model_num=3
  target_num=len(reference_states_cell)
  total_loss=0
  convergence_rate=0.7
  total_length=0


  for k in range(model_num):
    if k==0:
      Wm=np.loadtxt('W_0gram_tip.dat')
    if k==1:
      Wm=np.loadtxt('W_10gram_tip.dat')
    if k==2:
      Wm=np.loadtxt('W_20gram_tip.dat')


    for j in range(target_num):
      reference_states=reference_states_cell[j]
      target_points,state_dim=reference_states.shape
      total_length=total_length+target_points
 
      temp_state=initial_states_cell[j].reshape(1,state_dim)
      fai_value=contration_metric(temp_state)
      fai_value=tf.nn.sigmoid(fai_value)
      fai_value=tf.reshape(fai_value,(state_dim,state_dim))
      contration_matrix=tf.matmul(tf.transpose(fai_value), fai_value)+tf.eye(state_dim)
      temp_state_error=temp_state-reference_states[0,:]
      temp_state_error = tf.cast(temp_state_error, tf.float32) 
      distance_pre=tf.matmul(tf.matmul(temp_state_error,contration_matrix), tf.transpose(temp_state_error))
               
      for i in range(target_points):
        temp_target=reference_states[i,:].reshape(1,state_dim)
        control_signal=control_policy(tf.concat([temp_state, temp_target], 1))
        control_signal=tf.nn.sigmoid(control_signal)

        G=G_model(tf.concat([temp_state, control_signal], 1))
        G=tf.nn.sigmoid(G)
        temp_state=tf.matmul(G, Wm)

        temp_state_error=temp_state-temp_target
        tracking_loss=tf.reduce_sum(tf.square(temp_state_error)) 


        fai_value=contration_metric(temp_state)
        fai_value=tf.nn.sigmoid(fai_value)
        fai_value=tf.reshape(fai_value,(state_dim,state_dim))      
        contration_matrix=tf.matmul(tf.transpose(fai_value), fai_value)+tf.eye(state_dim)
        ccl=contraction_condition_loss(contration_matrix)

        distance_post=tf.matmul(tf.matmul(temp_state_error,contration_matrix), tf.transpose(temp_state_error))
        contraction_distance_loss=tf.nn.relu(distance_post-convergence_rate*distance_pre)
        contraction_loss=contraction_distance_loss+ccl

        total_loss=total_loss+tracking_loss+contraction_loss 
        distance_pre=distance_post
        

  return total_loss/total_length



