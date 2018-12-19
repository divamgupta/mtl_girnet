
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer

from SluiceUtils import *



class CrossStitch_SeqLab(Trainer):



    def build_model(self):

        config = self.config
        embed = Embedding( self.config['vocab_size']  ,  self.config['embed_dim']  , mask_zero=True)

        rnn_hi = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_en = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_enhi = (LSTM( self.config['nHidden'] , return_sequences=True ))
    
        if config['n_layers'] == 2:
            rnn_hi2 = (LSTM( self.config['nHidden'] , return_sequences=True ))
            rnn_en2 = (LSTM( self.config['nHidden'] , return_sequences=True ))
            rnn_enhi2 = (LSTM( self.config['nHidden'] , return_sequences=True ))

        stitch_layer = CrossStitch()
        stitch_layer.supports_masking  = True
        
        if config['n_layers'] == 2:
            stitch_layer2 = CrossStitch()
            stitch_layer2.supports_masking  = True
            
            
            
        def cal_cs_1l( inp ):
            x = embed(inp)
            x_hi = rnn_hi( x )

            # en 
            x = embed(inp)
            x_en = rnn_en( x )


            x = embed(inp)
            x_enhi = rnn_enhi( x )

            [ x_hi , x_en, x_enhi ] = stitch_layer([ x_hi , x_en , x_enhi ])

            return [ x_hi , x_en, x_enhi ]
        
        
        def cal_cs_2l( inp ):
            x = embed(inp)
            x_hi = rnn_hi( x )

            # en 
            x = embed(inp)
            x_en = rnn_en( x )


            x = embed(inp)
            x_enhi = rnn_enhi( x )

            [ x_hi , x_en, x_enhi ] = stitch_layer([ x_hi , x_en , x_enhi ])


            x_hi = rnn_hi2( x_hi )
            x_en = rnn_en2( x_en )
            x_enhi = rnn_enhi2( x_enhi )

            [ x_hi , x_en, x_enhi ] = stitch_layer2([ x_hi , x_en , x_enhi ])

            return [ x_hi , x_en, x_enhi ]
        
        
        if config['n_layers'] == 1:
            cal_cs = cal_cs_1l
        if config['n_layers'] == 2:
            cal_cs = cal_cs_2l
            
            
            
        # hi
        inp_hi = Input((self.config['sent_len'] , ))
        # en 
        inp_en = Input((self.config['sent_len'] , ))

        inp_enhi = Input((self.config['sent_len'] , ))

        [ x_hi , _ , _ ] = cal_cs( inp_hi)
        [ _ , x_en , _ ] = cal_cs( inp_en)
        [ _ , _ , x_enhi ] = cal_cs( inp_enhi)


        out_enhi = TimeDistributed(Dense( self.config['n_class_enhi']  , activation='softmax'))(x_enhi)
        out_hi = TimeDistributed(Dense( config['n_class_hi']   , activation='softmax'))(x_hi)

        out_en = TimeDistributed(Dense( config['n_class_en'] , activation='softmax'))(x_en)
        
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
config['n_layers'] = 1 # 1 or 2 


model = CrossStitch_SeqLab( exp_location="./ttt" , config_args = config )
model.train()


"""