from keras.models import *
from keras.layers import *

import keras
import keras.backend as K

# make sure the first n channels of input are the mask
# make sure that the scalar sum less than one!

# this layer will take in lstm cells as input and return an interleaved cell. 
# This can be than passed to the rnn class
# make sure all the aux cells have same state size
class InterleaveLSTMCells( keras.layers.Layer ) :

    # cells -> list of the lstm cells
    def __init__(self, cells, **kwargs):
        self.cells = cells
        self.n_hidden = cells[0].units
        self.state_size = [ self.n_hidden , self.n_hidden ]
        super( InterleaveLSTMCells , self).__init__(**kwargs)

    def build(self, input_shape):
        # the input should concatenated by scalar gates
        input_shape_n = ( input_shape[0] , input_shape[1]- len(self.cells ) )
        
        for cell in self.cells:
            self._trainable_weights += ( cell.trainable_weights )
        
        for cell in self.cells:
            self._non_trainable_weights += (  cell.non_trainable_weights )
        
        self.built = True

    def call(self, inputs, states):
        
        gate_vals = [ inputs[ : , i:i+1] for i in range(len(self.cells)) ]
        inputs  = inputs[ : , len(self.cells) : ]
        
        for i in range(len(self.cells)) : 
            gate_vals[i] = K.repeat_elements(gate_vals[i] , self.n_hidden , -1 ) 
        
        h_primaux = [None for  i in range(len(self.cells)) ]
        c_primaux = [None for  i in range(len(self.cells)) ]
        
        for  i in range(len(self.cells)):
            _ , [ h_primaux[i] , c_primaux[i] ]  = self.cells[i].call( inputs , states )
        
        gate_val_sum = 0
        for  i in range(len(self.cells)):
            
            if i == 0:
                h = gate_vals[i] * h_primaux[i]
                c = gate_vals[i] * c_primaux[i]
            else:
                h = h + gate_vals[i] * h_primaux[i]
                c = c + gate_vals[i] * c_primaux[i] 
            gate_val_sum += gate_vals[i]
        
        h = h  + (1 - gate_val_sum )*states[0]
        c = c  + (1 - gate_val_sum )*states[1]
        
        return h, [h , c ]
    
    
    
    
    
def GIRNet( prim_imput ,  aux_lstms   , bidirectional_gating=True , return_sequences=True ):
    
    aux_cells = [ l.cell for l in  aux_lstms ]
    cell_combined = InterleaveLSTMCells( aux_cells )
    
    if bidirectional_gating:
        prime_out = Bidirectional(LSTM(32 , return_sequences=True))( prim_imput )
    else:
        prime_out = LSTM(32 , return_sequences=True)( prim_imput )
        
    scalar_gates = TimeDistributed( Dense( 1+len(aux_lstms) , activation='softmax') )( prime_out )
    scalar_gates = Lambda(lambda x : x[... , 1: ])(scalar_gates)
    x = Concatenate(-1)([ scalar_gates , prim_imput ])
    out_interleaved = RNN(cell_combined  , return_sequences=True )( x )
    
    if not return_sequences:
        prime_out = Lambda( lambda x : x[: , -1 , : ])(prime_out)
        out_interleaved = Lambda( lambda x : x[: , -1 , : ])(out_interleaved)
    
    return scalar_gates , prime_out , out_interleaved