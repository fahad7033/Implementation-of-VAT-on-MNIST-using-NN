from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)


#%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import * 
from keras.layers import *
from sklearn.metrics import accuracy_score
import tensorflow as tf

#### Make the dataset

from sklearn import datasets
from sklearn.utils import shuffle
from keras.datasets import mnist



############################################################################
###########################################################################
nb_classes =10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

y_test_original = y_test
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

    
    
##################################################################
#############   Model With VAT    ################################
##################################################################
    
def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.log(p + 1e-16) - tf.log(q + 1e-16)), axis=1)


def make_unit_norm(x):
    return x/(tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), axis=1)), [-1, 1]) + 1e-16)





network = Sequential()
network.add( Dense(512 ,activation='relu', input_shape=(784,)))
network.add(Dropout(0.2))
network.add( Dense(512 ,activation='relu'))
network.add(Dropout(0.2))
network.add( Dense(10  ))

#Step 1
model_input = Input((784,))
p_logit = network( model_input )
p = Activation('softmax')( p_logit )

#Step 2
r = tf.random_normal(shape=tf.shape( model_input ))
r = make_unit_norm( r )
p_logit_r = network( model_input + 10*r  )

#Step 3 getting the adversarial perturbation
kl = tf.reduce_mean(compute_kld( p_logit , p_logit_r ))
grad_kl = tf.gradients( kl , [r ])[0]
r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm( r_vadv )/3.0

 #Step 4 getting the loss
p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_adv = network( model_input  + r_vadv )
vat_loss =  tf.reduce_mean(compute_kld( p_logit_no_gradient, p_logit_r_adv ))






### Training the model 
model_vat = Model(model_input , p )
model_vat.add_loss( vat_loss   )

model_vat.compile( loss='categorical_crossentropy' , optimizer='adadelta' ,  metrics=['accuracy'])

model_vat.metrics_names.append('vat_loss')
model_vat.metrics_tensors.append( vat_loss )



model_vat.fit(  X_train , y_train, batch_size=128 , epochs=12 , verbose=1 )

y_pred  = model_vat.predict( X_test ).argmax(-1)
print( "Test accruracy " , accuracy_score(y_test_original , y_pred  ))



