from keras.models import *
from keras.layers import *
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
import keras

from keras.initializers import Initializer
import tensorflow as tf


out_selector_initializer_biased = 'uniform' 



class OutSelctorInitBiases(Initializer):
    def __call__(self, shape, dtype=None):
        assert len( shape ) == 1
        m =  0.05 + np.zeros(shape)
        m[-1] = 0.95
        m = m.astype( np.float32 )
        m =  tf.convert_to_tensor( m )
        return m

    

class OutPutSelector(Layer):

    def __init__(self, **kwargs):        
        super(OutPutSelector, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type( input_shape ) is list # we need a list of inputs 
        for ish in input_shape:
            assert ish == input_shape[0]
                
        self.n_inputs = len( input_shape )
        
        self.betas = self.add_weight(name='betas', 
                                      shape=( self.n_inputs ,  ),
                                      initializer=OutSelctorInitBiases(),
                                      trainable=True)
        
        super( OutPutSelector , self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp ):
        for j in range(  self.n_inputs):
            if j == 0:
                out_i =  inp[j]*self.betas[ j ]
            else:
                out_i += inp[j]*self.betas[ j ]
        return out_i    
            
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    
    
    
from keras.initializers import Initializer


class CrossSelectInitBiased(Initializer):
    def __call__(self, shape, dtype=None):
        assert len( shape ) == 2
        m =  0.05*(1-np.identity(shape[0] ))  + 0.95*np.identity(shape[0] )
        m = m.astype( np.float32 )
        m =  tf.convert_to_tensor( m )
        return m


    
class CrossStitch(Layer):

    def __init__(self, **kwargs):        
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type( input_shape ) is list # we need a list of inputs 
        for ish in input_shape:
            assert ish == input_shape[0]
                
        self.n_inputs = len( input_shape )
        
        self.alphas = self.add_weight(name='alphas', 
                                      shape=( self.n_inputs , self.n_inputs ),
                                      initializer=CrossSelectInitBiased() ,
                                      trainable=True)
        
        super( CrossStitch , self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp ):
        outs = []
        for i in range( self.n_inputs ):
            for j in range(  self.n_inputs):
                if j == 0:
                    out_i =  inp[j]*self.alphas[i , j ]
                else:
                    out_i += inp[j]*self.alphas[i , j ]
            outs.append( out_i )
        return outs
    
    def compute_output_shape(self, input_shape):
        return input_shape
