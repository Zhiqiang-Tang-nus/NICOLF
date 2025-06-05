import tensorflow as tf


##----------------------------------------------------------------------------------------
##model structure (neural network)
class Robot_model():
    def __init__(self,dim_state_control,dim_W):
       self.model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=dim_state_control),
                                         tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                         tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                         tf.keras.layers.Dense(dim_W)])
    def __call__(self,model_input):
      return self.model(model_input)
    
    def model_param(self):
      return self.model.trainable_variables
    
    def save_model(self, filename):
      self.model.save(filename)

    def model_output(self,model_input):
      return self.model.predict(model_input)



##----------------------------------------------------------------------------------------
##controller structure (neural network)
class Control_policy():
  def __init__(self,dim_target_state,dim_output):
    self.model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=dim_target_state),
                                      tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                      tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                      tf.keras.layers.Dense(dim_output)])
    
  def __call__(self,model_input):
    return self.model(model_input)
    
  def model_param(self):
    return self.model.trainable_variables
    
  def save_model(self, filename):
    self.model.save(filename)

  def model_output(self,model_input):
    return self.model.predict(model_input)
        

##----------------------------------------------------------------------------------------
##contration metric structure (neural network)
class Contration_metric():
  def __init__(self,len_states):
    self.model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=len_states),
                                      tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                      tf.keras.layers.Dense(32, activation = tf.keras.activations.relu),
                                      tf.keras.layers.Dense(len_states*len_states)])

  def __call__(self,model_input):
    return self.model(model_input)
    
  def model_param(self):
    return self.model.trainable_variables
    
  def save_model(self, filename):
    self.model.save(filename)

  def model_output(self,model_input):
    return self.model.predict(model_input)
  

    


    