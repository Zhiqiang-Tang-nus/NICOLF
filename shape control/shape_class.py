import tensorflow as tf


class Shape_model():
  def __init__(self,input_dim,output_dim):
    self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation=tf.keras.activations.relu,input_shape=input_dim,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      tf.keras.layers.Conv2D(64,(3,3),activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      tf.keras.layers.Conv2D(64,(3,3),activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(64,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.sigmoid,kernel_initializer=tf.keras.initializers.HeNormal())])
    
    
  def __call__(self,model_input):
    return self.model(model_input)
    
  def model_param(self):
    return self.model.trainable_variables
    
  def save_model(self, filename):
    self.model.save(filename)

  def model_output(self,model_input):
    return self.model.predict(model_input)



class Shape_control_policy():
  def __init__(self,input_dim,output_dim):
    self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation=tf.keras.activations.relu,input_shape=input_dim,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      tf.keras.layers.Conv2D(32,(3,3),activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                      tf.keras.layers.Conv2D(32,(3,3),activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(32,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                      tf.keras.layers.Dense(output_dim,activation=tf.keras.activations.softmax,kernel_initializer=tf.keras.initializers.HeNormal())])
    
  def __call__(self,model_input):
    return self.model(model_input)
    
  def model_param(self):
    return self.model.trainable_variables
    
  def save_model(self, filename):
    self.model.save(filename)

  def model_output(self,model_input):
    return self.model.predict(model_input)
        


class Contraction_metric():
  def __init__(self,input_dim,output_dim):
      self.model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_dim),
                                        tf.keras.layers.Dense(128, activation = tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.HeNormal()),
                                        tf.keras.layers.Dense(64, activation = tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                        tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.HeNormal()),
                                        tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.sigmoid,kernel_initializer=tf.keras.initializers.HeNormal())])

  def __call__(self,model_input):
    return self.model(model_input)
    
  def model_param(self):
    return self.model.trainable_variables
    
  def save_model(self, filename):
    self.model.save(filename)

  def model_output(self,model_input):
    return self.model.predict(model_input)

    


    