import tensorflow as tf

def contraction_condition_loss(contration_matrix):
    eps=0.1
    m_lb = 1
    m_ub = 2
    state_dim=tf.shape(contration_matrix).numpy()
    state_dim=state_dim[0]

# # test positive definite of contration_matrix
    test_eigvals=tf.linalg.eigvals(contration_matrix-m_lb*tf.eye(state_dim)).numpy()
    min_eig = test_eigvals.real.min()     
    loss1 = tf.nn.relu(eps - min_eig)

    test_eigvals=tf.linalg.eigvals(m_ub*tf.eye(state_dim)-contration_matrix).numpy()
    min_eig = test_eigvals.real.min()    
    loss2 = tf.nn.relu(eps - min_eig)
    
    total_loss=loss1+loss2
    total_loss=tf.cast(total_loss, tf.float32)
    return total_loss
