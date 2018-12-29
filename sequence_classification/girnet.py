
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
    
    
    
    
class GIRNet_SeqClass(Trainer):


    def build_model(self):
        
        config = self.config
        emb = Embedding(  self.config['vocab_size'] , self.config['embed_dim']  )
        rnn_en = LSTM( self.config['nHidden'] )
        rnn_es = LSTM( self.config['nHidden'] )
        
        
        
        inp_en = Input(( self.config['sent_len']  ,))
        x_en = emb(inp_en )
        x_en = rnn_en( x_en )
        out_en = Dense( self.config['n_class_en']  , activation='softmax')( x_en )


        inp_es = Input(( self.config['sent_len']  ,))
        x_es = emb(inp_es )
        x_es = rnn_es( x_es )
        out_es = Dense( self.config['n_class_es']  , activation='softmax')( x_es )
        
        
        cell_en = rnn_en.cell
        cell_es = rnn_es.cell
        
        cell_combined = GiretTwoCell(cell_en , cell_es , self.config['nHidden']  )

        
        inp_enes = Input(( self.config['sent_len'] ,))
        x = emb(inp_enes )

        x_att = Bidirectional(LSTM(32 , return_sequences=True))( x )

        bider_last = Lambda( lambda x : x[: , -1 , : ])(x_att)

        x_att = TimeDistributed( Dense(3, activation='softmax') )(x_att)
        x_att = Lambda(lambda x : x[... , 1: ])(x_att)

        x = Concatenate(-1)([x_att , x ])

        x =  RNN(cell_combined )( x )

        x = Concatenate()([bider_last,x])

        out_enes = Dense( self.config['n_class_enes'] , activation='softmax')( x )

        self.model = Model([inp_en , inp_es , inp_enes] , [out_en , out_es , out_enes] )
        Trainer.build_model( self  )
        
        
        

        
        
        
        
# jjj
"""
config = {}
config['epochs'] = 2
config['dataset'] = "/tmp/senti_prepped.h5"

config['exp_name'] = 'pos_girnet_1l'
config['embed_dim'] = 300
config['vocab_size'] = 35000
config['nHidden'] = 64
config['sent_len'] = 50
config['n_class_en'] = 3
config['n_class_es'] = 3
config['n_class_enes'] = 3

model = GIRNet_SeqClass( exp_location="./ttt" , config_args = config )
model.train()


"""