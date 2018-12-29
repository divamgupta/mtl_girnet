
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer


class SharedPrivate_SeqClass(Trainer):


    def build_model(self):
        
        config = self.config
        embed = Embedding(  self.config['vocab_size'] , self.config['embed_dim']  )
        
        rnn_es = LSTM( self.config['nHidden']  )
        rnn_en = LSTM( self.config['nHidden']  )
        rnn_enes = LSTM( self.config['nHidden']  )
        if self.config['mode'] == 'parallel':
            rnn_shared = LSTM( self.config['nHidden']  , return_sequences=False  )
        else:
            rnn_shared = LSTM( self.config['nHidden']  , return_sequences=True  )
        
        
        
        inp_en = Input(( self.config['sent_len']  ,))
        x_en = embed(inp_en)
        if self.config['mode'] == 'parallel':
            
            x_en_1 = rnn_en( x_en )
            x_en_2 = rnn_shared( x_en )
            x_en = Concatenate()([ x_en_1 , x_en_2 ])
        else:
            x_en = rnn_en(  Concatenate()([ x_en , rnn_shared( x_en ) ])  )
            


        inp_es = Input(( self.config['sent_len']  ,))
        x_es = embed(inp_es)
        if self.config['mode'] == 'parallel':
            x_es_1 = rnn_es( x_es )
            x_es_2 = rnn_shared( x_es )
            x_es = Concatenate()([ x_es_1 , x_es_2 ])
        else:
            x_es = rnn_es(  Concatenate()([ x_es , rnn_shared( x_es ) ])  )
            
            
            
            
        
        inp_enes = Input(( self.config['sent_len'] ,))
        x_enes = embed( inp_enes )
        if self.config['mode'] == 'parallel':
            x_enes_1 = rnn_enes( x_enes )
            x_enes_2 = rnn_shared( x_enes )
            x_enes = Concatenate()([ x_enes_1 , x_enes_2 ])
        else:
            x_enes = rnn_enes(  Concatenate()([ x_enes , rnn_shared( x_enes ) ])  )
            
            
        out_enes = (Dense( self.config['n_class_enes'] , activation='softmax'))(x_enes)
        out_es = (Dense( self.config['n_class_es'] , activation='softmax'))(x_es)
        out_en = (Dense( self.config['n_class_en'] , activation='softmax'))(x_en)

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
config['mode'] = "parallel" # stacked , parallel

model = SharedPrivate_SeqClass( exp_location="./ttt" , config_args = config )
model.train()


"""