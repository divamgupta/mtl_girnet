from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf

from Utils import Trainer
from att_functions import *



class LowSupShare_ABSA(Trainer):
    """docstring for LowSupShare_ABSA"""


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
        m.load_weights( "./data/lr_lstm_glove_3.5_42B_ep1.h5"  )

        return rnn_left.cell.get_weights() , rnn_right.cell.get_weights()


    def getAuxM(self ):

        inp = Input((  self.config['maxSentenceL'] ,  ) ) # left


        inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp) # right

        [ left_x_1 , right_x_1 , left_x_2, right_x_2  ]  = self.cal_cs( inp , inp_rev )
        right_x_1  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x_1 )

        c_x = Concatenate( axis=-1 )([ left_x_1 ,right_x_1] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )

        return [inp] , [out]


    def getPrimM( self ):

        left_i = Input(( self.config['maxSentenceL'] ,  ) )
        right_i = Input(( self.config['maxSentenceL'] ,  ) )
        tar_i = Input(( self.config['maxTarLen']  ,  ) )


        [ left_x_1 , right_x_1 , left_x_2, right_x_2  ]  = self.cal_cs( left_i , right_i )

        sent_len_l = Input((1,) )
        sent_len_r = Input((1,) )
        sent_len_t = Input((1,) )


        tar_x = self.glove_embed42( tar_i  )
        tar_x = Dropout(0.5)( tar_x  )



        hidden_l = left_x_2
        hidden_r = right_x_2
        hidden_t = Bidirectional( GRU( self.config['nHidden']  , return_sequences=True) )( tar_x  )


        pool_t = Lambda( lambda x:reduce_mean_with_len( x[0] , x[1] , padding='right'))([ hidden_t , sent_len_t  ])


        skip_cell_weights_left = TimeDistributed( Dense( 1
                   , bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None) 
                   , kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1 , seed=None) 
                   , use_bias=True))(  hidden_l )

        skip_cell_weights_right = TimeDistributed( Dense( 1
                   , bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None) 
                   , kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1 , seed=None) 
                   , use_bias=True))(  hidden_r )


        skip_cell_weights_left = Reshape(( self.config['maxSentenceL'] , ) )( skip_cell_weights_left )
        skip_cell_weights_left = Lambda( lambda x: sigmoid_with_len_l( x[0] , x[1] , self.config['maxSentenceL'] , padding='right' ) )([ skip_cell_weights_left , sent_len_l ])
        skip_cell_weights_left = Reshape(( self.config['maxSentenceL'] ,1 ) )( skip_cell_weights_left )

        skip_cell_weights_right = Reshape(( self.config['maxSentenceL'] , ) )( skip_cell_weights_right )
        skip_cell_weights_right = Lambda( lambda x: sigmoid_with_len_l( x[0] , x[1] , self.config['maxSentenceL'] , padding='right' ) )([ skip_cell_weights_right , sent_len_r ])
        skip_cell_weights_right = Reshape(( self.config['maxSentenceL'] ,1 ) )( skip_cell_weights_right )

        pool_right_prim = reduce_via_attention( hidden_r , skip_cell_weights_right , mean=True)
        pool_left_prim = reduce_via_attention( hidden_l , skip_cell_weights_left , mean=True)



        hidden_l_aux = left_x_1
        hidden_r_aux = right_x_1


        pool_right_aux = reduce_via_attention( hidden_r_aux , skip_cell_weights_right , mean=True)
        pool_left_aux = reduce_via_attention( hidden_l_aux , skip_cell_weights_left , mean=True)


        feats = Concatenate()([  pool_left_prim , pool_right_prim , pool_right_aux, pool_left_aux , pool_t ])
        feats_juss_aux = Concatenate()([   pool_right_aux, pool_left_aux  ])

        out = Dense(3 , activation='softmax' )( feats  )
        out_just_aux = Dense(3 , activation='softmax' )( feats_juss_aux  )

        return [left_i , right_i , tar_i ,sent_len_l ,sent_len_r , sent_len_t ] ,  [out,out_just_aux]



    
    
    def cal_cs( self ,  inp_l , inp_r ):
    
        left_x = self.glove_embed42( inp_l )
        right_x = self.glove_embed42( inp_r )

        left_x_1 = self.rnn_left_( left_x )
        right_x_1 = self.rnn_right_( right_x )

        left_x_2 = self.rnn_left_2( left_x_1 )
        right_x_2 = self.rnn_right_2( right_x_1 )

        return [ left_x_1 , right_x_1 , left_x_2, right_x_2  ]


    def build_model(self):

        self.get_glove()
        
        self.rnn_left_ = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout'] , trainable=True  )
        self.rnn_right_ = GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout']  , trainable=True )

        self.rnn_left_2 = ( GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout'] , trainable=True  ))
        self.rnn_right_2 = ( GRU( self.config['nHidden'] , return_sequences=True , dropout=self.config['dropout'] , recurrent_dropout=self.config['recurrent_dropout']  , trainable=True ))
        
        aux_inps , aux_outs = self.getAuxM()
        prim_inps , prim_outs = self.getPrimM()

        self.model = Model( prim_inps+aux_inps , prim_outs+aux_outs )
        Trainer.build_model( self  )
        
"""
        
config = {}
config['epochs'] = 4
config['dataset'] = "../../data/prepped/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
# config['dataset'] =  "../../data/prepped/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
config['maxSentenceL'] = 80
config['maxTarLen'] = 10
config['nHidden'] = 64
config['dropout'] = 0.2
config['recurrent_dropout'] = 0.2

model = LowSupShare_ABSA( exp_location="./ttt" , config_args = config )
model.train()

"""