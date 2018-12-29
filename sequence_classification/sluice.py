
from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer


from SluiceUtils import *


class Sluice_SeqClass(Trainer):


    def build_model(self):
        
        config = self.config
        
        embed = Embedding(  self.config['vocab_size'] , self.config['embed_dim']  )

        rnn_hi = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_en = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_enhi = (LSTM( self.config['nHidden'] , return_sequences=True ))

        rnn_hi2 = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_en2 = (LSTM( self.config['nHidden'] , return_sequences=True ))
        rnn_enhi2 = (LSTM( self.config['nHidden'] , return_sequences=True ))

        stitch_layer = CrossStitch()
        stitch_layer.supports_masking  = True

        osel = OutPutSelector()
        osel.supports_masking  = True
        
        
        

        def desectOut(xx):
            l = xx.shape[-1]
            return Lambda( lambda x : [x[ ... , :l/2 ] , x[ ... , l/2: ] ] )( xx )
        
        
        def cal_cs( inp ):
            x = embed(inp)
            x_hi = rnn_hi( x )

            # en 
            x = embed(inp)
            x_en = rnn_en( x )


            x = embed(inp)
            x_enhi = rnn_enhi( x )


            [ x_hi1 , x_hi2 ] = desectOut( x_hi )
            [ x_en1 , x_en2 ] = desectOut( x_en )
            [ x_enhi1 , x_enhi2 ] = desectOut( x_enhi )

            [ x_hi1 , x_en1 , x_enhi1 , x_hi2 , x_en2 , x_enhi2 ] = stitch_layer([ x_hi1 , x_en1 , x_enhi1 , x_hi2 , x_en2 , x_enhi2 ])

            x_hi =  Concatenate()([  x_hi1 , x_hi2 ])
            x_en =  Concatenate()([  x_en1 , x_en2 ])
            x_enhi =  Concatenate()([  x_enhi1 , x_enhi2 ])

            x_hi_p = x_hi
            x_en_p = x_en
            x_enhi_p = x_enhi

            x_hi = rnn_hi2( x_hi )
            x_en = rnn_en2( x_en )
            x_enhi = rnn_enhi2( x_enhi )

            [ x_hi1 , x_hi2 ] = desectOut( x_hi )
            [ x_en1 , x_en2 ] = desectOut( x_en )
            [ x_enhi1 , x_enhi2 ] = desectOut( x_enhi )

            [ x_hi1 , x_en1 , x_enhi1 , x_hi2 , x_en2 , x_enhi2 ] = stitch_layer([ x_hi1 , x_en1 , x_enhi1 , x_hi2 , x_en2 , x_enhi2 ])

            x_hi =  Concatenate()([  x_hi1 , x_hi2 ])
            x_en =  Concatenate()([  x_en1 , x_en2 ])
            x_enhi =  Concatenate()([  x_enhi1 , x_enhi2 ])

            x_hi  = osel([ x_hi ,  x_hi_p ])
            x_en  = osel([ x_en ,  x_en_p ])
            x_enhi  = osel([ x_enhi ,  x_enhi_p ])


            return [ x_hi , x_en, x_enhi ]

        
        inp_en = Input(( self.config['sent_len']  ,))
        inp_es = Input(( self.config['sent_len']  ,))
        inp_enes = Input(( self.config['sent_len'] ,))
        
        [ x_es , _ , _ ] = cal_cs( inp_es )
        [ _ , x_en , _ ] = cal_cs( inp_en )
        [ _ , _ , x_enes ] = cal_cs( inp_enes )
        
        
        x_es = Lambda( lambda x: x[: , -1])( x_es )
        x_en = Lambda( lambda x: x[: , -1])( x_en )
        x_enes = Lambda( lambda x: x[: , -1])( x_enes )

        
            
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
config['nHidden'] = 100
config['sent_len'] = 50
config['n_class_en'] = 3
config['n_class_es'] = 3
config['n_class_enes'] = 3
config['n_layers'] = 1 # 1 , 2 

model = Sluice_SeqClass( exp_location="./ttt" , config_args = config )
model.train()


"""