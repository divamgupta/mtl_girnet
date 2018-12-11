from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf

from Utils import Trainer


from att_functions import *


class HardShare_ABSA(Trainer):
    """docstring for GIRNet_ABSA"""


    def get_glove(self):

        import h5py
        gf = h5py.File(glove_path)
        gloveVecs_42 = np.array( gf['glove_common_42_vecs'] )
        gloveSize  = gloveVecs_42.shape[-1]
        vocabSize = gloveVecs_42.shape[0]
        
        self.gloveSize = gloveSize
        self.vocabSize = vocabSize

        self.glove_embed42 = (Embedding( vocabSize , gloveSize , weights=[gloveVecs_42] , trainable=False ))




    def get_document_level_rnn_weights(self):

        inp = Input(( self.config['maxSentenceL'] ,  ) ) # left
        inp_x = inp

        embed = (Embedding( self.vocabSize , self.gloveSize ,   trainable=False )  )

        inp_x = self.glove_embed42 ( inp_x )

        inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp_x) # right

        rnn_left = GRU( 64 , return_sequences=True , dropout=self.config['dropout']  , recurrent_dropout=self.config['recurrent_dropout']  )
        rnn_right = GRU( 64 , return_sequences=True , dropout=self.config['dropout']  , recurrent_dropout=self.config['recurrent_dropout']  )

        left_x = self.rnn_left( inp_x )
        right_x = self.rnn_right( inp_rev  )
        right_x  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x )

        c_x = Concatenate( axis=-1 )([ left_x ,right_x] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )

        m = Model( inp  , out )
        m.load_weights("./data/lr_lstm_glove_3.5_42B_ep1.h5"  )

        return rnn_left.cell.get_weights() , rnn_right.cell.get_weights()


    def getAuxM( self ):

        inp = Input(( self.config['maxSentenceL'] ,  ) ) # left
        inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp) # right
        
        inpx = self.glove_embed42(inp )
        inp_rev = self.glove_embed42(inp_rev )
        
        
        if self.config['n_layers'] == 2:
            left_x_1 = self.rnn_left_2(  self.rnn_left_(inpx))
            right_x_1 =  self.rnn_right_2( self.rnn_right_(inp_rev))
        else:
            left_x_1 = self.rnn_left_(inpx)
            right_x_1 = self.rnn_right_(inp_rev)

        right_x_1  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x_1 )

        c_x = Concatenate( axis=-1 )([ left_x_1 ,right_x_1] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )
        
        return [inp] , [out]


    def getPrimM(self):

        left_i = Input(( self.config['maxSentenceL'] ,  ) )
        right_i = Input(( self.config['maxSentenceL'] ,  ) )
        tar_i = Input(( self.config['maxTarLen'] ,  ) )
        
        if not self.config['n_layers'] == 2:    
            left_x_1 = self.rnn_left_( self.glove_embed42( left_i ))
            right_x_1 = self.rnn_right_( self.glove_embed42( right_i  ))
        else:
            left_x_1 = self.rnn_left_2(  self.rnn_left_(  self.glove_embed42( left_i )))
            right_x_1 =  self.rnn_right_2( self.rnn_right_(  self.glove_embed42( right_i  )))

        sent_len_l = Input((1,) )
        sent_len_r = Input((1,) )
        sent_len_t = Input((1,) )


        tar_x = self.glove_embed42( tar_i  )
        tar_x = Dropout(0.5)( tar_x  )

        hidden_t = Bidirectional( GRU( self.config['nHidden'] , return_sequences=True) )( tar_x  )

        
        left_pool = Lambda( lambda x:reduce_mean_with_len( x[0] , x[1] , padding='right'))([ left_x_1 , sent_len_l  ])
        right_pool = Lambda( lambda x:reduce_mean_with_len( x[0] , x[1] , padding='right'))([ right_x_1 , sent_len_r  ])

        feats = Concatenate()([ left_pool , right_pool  ])
        feats_juss_aux = Concatenate()([left_pool , right_pool ])

        return [left_i , right_i , tar_i ,sent_len_l ,sent_len_r , sent_len_t ] ,  [ Dense(3 , activation='softmax' )( feats  ) ,  Dense(3 , activation='softmax' )( feats  )  ]



    def build_model(self):

        self.get_glove()

        self.rnn_left_ = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout'] , trainable=True  )
        self.rnn_right_ = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout']  , trainable=True )
        
        if self.config['n_layers'] == 2:
            self.rnn_left_2 = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout'] , trainable=True  )
            self.rnn_right_2 = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout']  , trainable=True )
        
        
        aux_inps , aux_outs = self.getAuxM()
        prim_inps , prim_outs = self.getPrimM()
        self.model = Model( prim_inps+aux_inps , prim_outs+aux_outs )
        Trainer.build_model( self  )

        
        
"""
        
config = {}
config['epochs'] = 4
#config['dataset'] = "../../data/prepped/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
config['dataset'] =  "../../data/prepped/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
config['maxSentenceL'] = 80
config['maxTarLen'] = 10
config['nHidden'] = 64
config['dropout'] = 0.2
config['recurrent_dropout'] = 0.2
config['n_layers'] = 2

model = HardShare_ABSA( exp_location="./ttt" , config_args = config )
model.train()

"""