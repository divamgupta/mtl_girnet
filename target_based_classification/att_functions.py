from keras.models import *
from keras.layers import *
import keras
from myutils.keras_utils import allow_growth , showKerasModel
allow_growth()
from myutils.pyutils import env_arg
import tensorflow as tf


glove_path = "../data/prepped/glovePrepped.h5"

def sigmoid_with_len_l( inputs , length, max_len , padding='right' ):
    assert len( inputs.shape) == 2
    if padding == 'left':
        inputs = K.reverse(inputs , -1 )
        
    inputs = tf.cast(inputs, tf.float32)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs = tf.sigmoid( inputs )
    inputs *= mask
    ret = inputs
    if padding == 'left':
        ret = K.reverse(ret , -1 )

    return ret


def softmax_with_len_l( inputs , length, max_len , padding='right' ):
    assert len( inputs.shape) == 2
    if padding == 'left':
        inputs = K.reverse(inputs , -1 )
        
    inputs = tf.cast(inputs, tf.float32)
    max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    inputs = tf.exp(inputs - max_axis)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    ret =  inputs / _sum

    if padding == 'left':
        ret = K.reverse(ret , -1 )

    return ret



def reduce_mean_with_len(inputs, length , padding='right' ):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    
    if padding == 'left':
        inputs = K.reverse(inputs , 1 )
        
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    
    return inputs



def reduce_via_attention( inputs , att_vals , mean=True):
    assert len( att_vals.shape ) == 3 and att_vals.shape[-1] == 1
    max_len = int(att_vals.shape[-1])
#     att_vals = Reshape(( max_len ,1 ) )( att_vals )
    att_vals = Lambda(lambda x: K.repeat_elements(x, int( inputs.shape[-1])  , -1 ) )(att_vals)
    
    att_mem_mul = Multiply()([att_vals , inputs ])
    attented_vec_sum_num = Lambda(lambda x: K.sum(x,1) )(att_mem_mul  )
    
    if mean:
        sum_den = Lambda(lambda x: K.sum(x,1) )( att_vals  )
        attented_vec = Lambda( lambda x : tf.divide(x[0],x[1]+0.0000001  ))([attented_vec_sum_num,sum_den])
    else:
        attented_vec = attented_vec_sum_num
        
    return attented_vec

