
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer


class HardShare_SeqClass(Trainer):


    def build_model(self):
        
        config = self.config
        emb = Embedding(  self.config['vocab_size'] , self.config['embed_dim']  )
        
        if self.config['n_layers'] == 2:
            rnn = LSTM( self.config['nHidden'], return_sequences=True   )
            rnn2 = LSTM( self.config['nHidden']  )
        else:
            rnn = LSTM( self.config['nHidden'] )
        
        
        
        inp_en = Input(( self.config['sent_len']  ,))
        x = emb(inp_en)
        if self.config['n_layers'] == 2:
            x =rnn2( rnn( x ))
        else:
            x = rnn( x )
        out_en = Dense( self.config['n_class_en']  , activation='softmax')( x )


        inp_es = Input(( self.config['sent_len']  ,))
        x = emb(inp_es)
        if self.config['n_layers'] == 2:
            x =rnn2( rnn( x ))
        else:
            x = rnn( x )
        out_es = Dense( self.config['n_class_es']  , activation='softmax')( x )
        
        
        
        inp_enes = Input(( self.config['sent_len'] ,))
        x = emb(inp_enes)
        if self.config['n_layers'] == 2:
            x =rnn2( rnn( x ))
        else:
            x = rnn( x )
        out_enes = Dense( self.config['n_class_enes'] , activation='softmax')( x )

        self.model = Model([inp_en , inp_es , inp_enes] , [out_en , out_es , out_enes] )
        Trainer.build_model( self  )
        
        
        

        
        
        
        
# jjj

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
config['n_layers'] = 2

model = HardShare_SeqClass( exp_location="./ttt" , config_args = config )
model.train()


