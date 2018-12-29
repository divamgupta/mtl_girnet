
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer


from SluiceUtils import *


class CrossStitch_SeqClass(Trainer):


    def build_model(self):
        
        config = self.config
        
        
        embed = Embedding(  self.config['vocab_size'] , self.config['embed_dim']  )

        
        
        if self.config['n_layers'] == 2:
            rnn_es = LSTM( self.config['nHidden']  , return_sequences=True )
            rnn_en = LSTM( self.config['nHidden']  , return_sequences=True )
            rnn_enes = LSTM( self.config['nHidden'] , return_sequences=True )
        
            rnn_es2 = LSTM( self.config['nHidden'] )
            rnn_en2 = LSTM( self.config['nHidden'] )
            rnn_enes2 = LSTM( self.config['nHidden'] )
        else:
            rnn_es = LSTM( self.config['nHidden']  )
            rnn_en = LSTM( self.config['nHidden']  )
            rnn_enes = LSTM( self.config['nHidden']  )

        stitch_layer = CrossStitch()
        stitch_layer.supports_masking  = True
        
        if self.config['n_layers'] == 2:
            stitch_layer2 = CrossStitch()
            stitch_layer2.supports_masking  = True
            
            
            
        def cal_cs1l( inp ):
            x = embed(inp)
            x_es = rnn_es( x )

            # en 
            x = embed(inp)
            x_en = rnn_en( x )


            x = embed(inp)
            x_enes = rnn_enes( x )

            [ x_es , x_en, x_enes ] = stitch_layer([ x_es , x_en , x_enes ])

            return [ x_es , x_en, x_enes ]
        
        def cal_cs2l( inp ):
            x = embed(inp)
            x_es = rnn_es( x )

            # en 
            x = embed(inp)
            x_en = rnn_en( x )


            x = embed(inp)
            x_enes = rnn_enes( x )

            [ x_es , x_en, x_enes ] = stitch_layer([ x_es , x_en , x_enes ])


            x_es = rnn_es2( x_es )
            x_en = rnn_en2( x_en )
            x_enes = rnn_enes2( x_enes )

            [ x_es , x_en, x_enes ] = stitch_layer2([ x_es , x_en , x_enes ])

            return [ x_es , x_en, x_enes ]
        
        
        if self.config['n_layers'] == 2:
            cal_cs = cal_cs2l
        else:
            cal_cs = cal_cs1l


        
        inp_en = Input(( self.config['sent_len']  ,))
        inp_es = Input(( self.config['sent_len']  ,))
        inp_enes = Input(( self.config['sent_len'] ,))
        
        [ x_es , _ , _ ] = cal_cs( inp_es )
        [ _ , x_en , _ ] = cal_cs( inp_en )
        [ _ , _ , x_enes ] = cal_cs( inp_enes )
        
            
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
config['n_layers'] = 1 # 1 , 2 

model = CrossStitch_SeqClass( exp_location="./ttt" , config_args = config )
model.train()


"""