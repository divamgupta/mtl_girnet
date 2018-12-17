
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer




class HardShare_SeqLab(Trainer):



    def build_model(self):


        embed = Embedding( self.config['vocab_size']  ,  self.config['embed_dim']  , mask_zero=True)
        
        rnn = (LSTM( self.config['nHidden'] , return_sequences=True ))
        if config['n_layers'] == 2:
            rnn2 = (LSTM( self.config['nHidden'] , return_sequences=True ))

        # hi

        inp_hi = Input(( self.config['sent_len'] , ))
        x = embed(inp_hi)
        if config['n_layers'] == 2:
            rnn2(rnn( x ))
        else:
            x = rnn( x )
        out_hi = TimeDistributed(Dense( config['n_class_hi'] , activation='softmax'))(x)
        
        
        # en

        inp_en = Input(( self.config['sent_len'] , ))
        x = embed(inp_en)
        if config['n_layers'] == 2:
            rnn2(rnn( x ))
        else:
            x = rnn( x )
        out_en = TimeDistributed(Dense( config['n_class_en']  , activation='softmax'))(x)


        
        inp_enhi = Input(( self.config['sent_len'] , ))
        x = embed(inp_enhi)
        if config['n_layers'] == 2:
            rnn2(rnn( x ))
        else:
            x = rnn( x )
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
config['n_layers'] = 1

model = HardShare_SeqLab( exp_location="./ttt" , config_args = config )
model.train()


"""