
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer


class GiretTwoCell(keras.layers.Layer):

    def __init__(self, cell_1 , cell_2 , nHidden , **kwargs):
        self.cell_1 = cell_1
        self.cell_2 = cell_2
        self.nHidden = nHidden
        self.state_size = [nHidden,nHidden]
        super(GiretTwoCell, self).__init__(**kwargs)

    def build(self, input_shape):
        
        nHidden = self.nHidden
        
        input_shape_n = ( input_shape[0] , input_shape[1]- 2 )
#         print "pp", input_shape_n
        
#         self.cell_1.build(input_shape_n)
#         self.cell_2.build(input_shape_n)
        
        self._trainable_weights += ( self.cell_1.trainable_weights )
        self._trainable_weights += ( self.cell_2.trainable_weights )
        
        self._non_trainable_weights += (  self.cell_1.non_trainable_weights )
        self._non_trainable_weights += (  self.cell_2.non_trainable_weights )
        
        self.built = True

    def call(self, inputs, states):
        
        nHidden = self.nHidden
        
        gate_val_1 = inputs[ : , 0:1]
        gate_val_2 = inputs[ : , 1:2]
        
        inputs  = inputs[ : , 2: ]
                
        gate_val_1 = K.repeat_elements(gate_val_1 , nHidden , -1 ) # shape # bs , hidden
        gate_val_2 = K.repeat_elements(gate_val_2 , nHidden , -1 ) # shape # bs , hidden
        
        _ , [h1 , c1 ]  = self.cell_1.call( inputs , states )
        _ , [h2 , c2 ]  = self.cell_2.call( inputs , states )
        
        h = gate_val_1*h1 + gate_val_2*h2  + (1 - gate_val_1 -  gate_val_2 )*states[0]
        c = gate_val_1*c1 + gate_val_2*c2  + (1 - gate_val_1 -  gate_val_2 )*states[1]
        
        return h, [h , c ]
    
    
    
    
class GIRNet_SeqLab(Trainer):


    def build_model(self):


        embed = Embedding( self.config['vocab_size']  ,  self.config['embed_dim']  , mask_zero=True)

        rnn_hi = LSTM( self.config['nHidden'] , return_sequences=True )
        rnn_en = LSTM( self.config['nHidden'] , return_sequences=True )
        
        # en

        inp_en = Input(( self.config['sent_len'] , ))
        x = embed(inp_en)
        x = rnn_en( x )
        out_en = TimeDistributed(Dense( config['n_class_en']  , activation='softmax'))(x)


        # hi

        inp_hi = Input(( self.config['sent_len'] , ))
        x = embed(inp_hi)
        x = rnn_hi( x )
        out_hi = TimeDistributed(Dense( config['n_class_hi'] , activation='softmax'))(x)
        cell_combined = GiretTwoCell( rnn_hi.cell , rnn_en.cell , self.config['nHidden'] )

        
        inp_enhi = Input(( self.config['sent_len'] , ))
        x = embed(inp_enhi )

        x_att = x
        x_att = Bidirectional(LSTM(32 , return_sequences=True))( x )
        bider_h = x_att 
        x_att = TimeDistributed( Dense(3, activation='softmax') )(x_att)
        x_att = Lambda(lambda x : x[... , 1: ])(x_att)

        x = Concatenate(-1)([x_att , x ])

        x =  RNN(cell_combined , return_sequences=True )( x )
        out_enhi = TimeDistributed(Dense( self.config['n_class_enhi'] , activation='softmax'))(x)
        
        self.model = Model( [inp_hi , inp_en , inp_enhi  ] , [ out_hi , out_en , out_enhi ] )
        Trainer.build_model( self  )
        
        
        

        
        
        
        
# jjj
"""
config = {}
config['epochs'] = 4
config['dataset'] = "/tmp/postag_prepped.h5"

config['exp_name'] = 'pos_girnet_1l'
config['embed_dim'] = 50
config['vocab_size'] = 30003
config['nHidden'] = 100
config['sent_len'] = 150
config['n_class_en'] = 45
config['n_class_hi'] = 25
config['n_class_enhi'] = 19

model = GIRNet_SeqLab( exp_location="./ttt" , config_args = config )
model.train()


"""