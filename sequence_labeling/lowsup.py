
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer





class LowSup_SeqLab(Trainer):



    def build_model(self):


        embed = Embedding( self.config['vocab_size']  ,  self.config['embed_dim']  , mask_zero=True)
        
        
        if self.config['mode'] == 'share':
            rnn_hi = (LSTM(  self.config['nHidden']  , return_sequences=True ))
            rnn_enhi = (LSTM(  self.config['nHidden']  , return_sequences=True ))
        elif self.config['mode'] == 'concat':
            rnn_en = (LSTM(  self.config['nHidden']  , return_sequences=True ))
            rnn_hi = (LSTM(  self.config['nHidden']  , return_sequences=True ))
            rnn_enhi = (LSTM(  self.config['nHidden']  , return_sequences=True ))
            
        # hi

        inp_hi = Input(( self.config['sent_len'] , ))
        x = embed(inp_hi)
        if self.config['mode'] == 'share':
            x = rnn_hi( x )
        if self.config['mode'] == 'concat':
            x = rnn_hi( x )
        out_hi = TimeDistributed(Dense( config['n_class_hi'] , activation='softmax'))(x)
        
        
        # en

        inp_en = Input(( self.config['sent_len'] , ))
        x = embed(inp_en)
        if self.config['mode'] == 'share':
            x = rnn_hi( x )
        if self.config['mode'] == 'concat':
            x = rnn_hi( x )
        out_en = TimeDistributed(Dense( config['n_class_en']  , activation='softmax'))(x)


        
        inp_enhi = Input(( self.config['sent_len'] , ))
        x = embed(inp_enhi)
        
        if self.config['mode'] == 'share':
            x_en = rnn_hi( x )
            x =  x_en 
            x = rnn_enhi( x )
        if self.config['mode'] == 'concat':
            x_en = rnn_en( x )
            x_hi = rnn_hi( x )
            x = Concatenate(-1)([ x_en , x_hi ])
            x = rnn_enhi( x )
        
        out_enhi = TimeDistributed(Dense(  self.config['n_class_enhi'] , activation='softmax'))(x)

        
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
config['mode'] = "concat" # concat or share


model = LowSup_SeqLab( exp_location="./ttt" , config_args = config )
model.train()


"""