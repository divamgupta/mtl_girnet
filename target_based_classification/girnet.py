from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf
from Utils import Trainer

from att_functions import *

def sigmoid_with_len_l( inputs , length, max_len , padding='right' ):

    assert len( inputs.shape) == 2
    if padding == 'left':
        inputs = K.reverse(inputs , -1 )

    inputs = tf.cast(inputs, tf.float32)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs = tf.sigmoid( inputs )
    inputs *= mask
    ret = inputs
    if padding == 'left':
        ret = K.reverse(ret , -1 )

    return ret


def reduce_mean_with_len(inputs, length , padding='right' ):


    if padding == 'left':
        inputs = K.reverse(inputs , 1 )

    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length

    return inputs



def reduce_mean_weighted( inputs , weight_vals , mean=True):
    assert len( weight_vals.shape ) == 3 and weight_vals.shape[-1] == 1
    max_len = int(weight_vals.shape[-1])
    weight_vals = Lambda(lambda x: K.repeat_elements(x, int( inputs.shape[-1])  , -1 ) )(weight_vals)

    wgt_mem_mul = Multiply()([weight_vals , inputs ])
    wgt_vec_sum_num = Lambda(lambda x: K.sum(x,1) )(wgt_mem_mul  )

    if mean:
        sum_den = Lambda(lambda x: K.sum(x,1) )( weight_vals  )
        attented_vec = Lambda( lambda x : tf.divide(x[0],x[1]+0.0000001  ))([wgt_vec_sum_num,sum_den])
    else:
        attented_vec = wgt_vec_sum_num

    return attented_vec




class SkipStepCell(keras.layers.Layer):

    def __init__(self, cell_1 , state_size , **kwargs):
        self.cell_1 = cell_1
        self.state_size = state_size
        super(SkipStepCell, self).__init__(**kwargs)

    def build(self, input_shape):

        input_shape_n = ( input_shape[0] , input_shape[1]-1 )
#         print "pp", input_shape_n

        if not self.cell_1.built:
            self.cell_1.build(input_shape_n)

        self._trainable_weights += ( self.cell_1.trainable_weights )
        self._non_trainable_weights += (  self.cell_1.non_trainable_weights )


        self.built = True

    def call(self, inputs, states):

        gate_val = inputs[ : , 0:1]
        inputs  = inputs[ : , 1: ]

        prev_output = states[0]

        gate_val = K.repeat_elements(gate_val , self.state_size , -1 ) # shape # bs , hidden

        tmp_output , _  = self.cell_1.call( inputs , [prev_output ] )
        output = gate_val*tmp_output + (1-gate_val)*prev_output
        return output, [output]






class GIRNet_ABSA(Trainer):
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

        left_x = rnn_left( inp_x )
        right_x = rnn_right( inp_rev  )
        right_x  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x )

        c_x = Concatenate( axis=-1 )([ left_x ,right_x] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )

        m = Model( inp  , out )
        m.load_weights( "./data/lr_lstm_glove_3.5_42B_ep1.h5" )

        return rnn_left.cell.get_weights() , rnn_right.cell.get_weights()


    def getAuxM(self):

        inp = Input(( self.config['maxSentenceL'] ,  ) ) # left
        inp_x = inp

        embed = (Embedding( self.vocabSize , self.gloveSize  ,   trainable=False )  )

        inp_x = self.glove_embed42 ( inp_x )

        inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp_x) # right

        rnn_left = RNN( self.rnn_left_aux_cell__ , return_sequences=True)
        rnn_right = RNN( self.rnn_right_aux_cell__ , return_sequences=True)

        left_x = rnn_left( inp_x )
        right_x = rnn_right( inp_rev  )
        right_x  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x )

        c_x = Concatenate( axis=-1 )([ left_x ,right_x] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )

        self.rnn_left_aux_cell__.set_weights(self.aux_w_l)
        self.rnn_right_aux_cell__.set_weights(self.aux_w_r)

        return [inp] , [out]




    def getPrimM(self):

        left_i = Input(( self.config['maxSentenceL'] ,  ) )
        right_i = Input(( self.config['maxSentenceL'] ,  ) )
        tar_i = Input(( self.config['maxTarLen'] ,  ) )

        sent_len_l = Input((1,) )
        sent_len_r = Input((1,) )
        sent_len_t = Input((1,) )


        left_x = self.glove_embed42( left_i  )
        right_x = self.glove_embed42( right_i  )
        tar_x = self.glove_embed42( tar_i  )

        left_x = Dropout(0.5)( left_x  )
        right_x = Dropout(0.5)( right_x  )
        tar_x = Dropout(0.5)( tar_x  )



        rnn_left_aux_cell = SkipStepCell(self.rnn_left_aux_cell__ , state_size=self.config['nHidden'] )
        rnn_right_aux_cell = SkipStepCell( self.rnn_right_aux_cell__  , state_size=self.config['nHidden'] )

        rnn_left_aux = RNN( rnn_left_aux_cell , return_sequences=True)
        rnn_right_aux = RNN( rnn_right_aux_cell , return_sequences=True)


        hidden_l = Bidirectional( GRU( self.config['nHidden'] , return_sequences=True) )( left_x  )
        hidden_r = Bidirectional( GRU( self.config['nHidden'] , return_sequences=True) )( right_x  )
        hidden_t = Bidirectional( GRU( self.config['nHidden'] , return_sequences=True) )( tar_x  )


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

        pool_right_prim = reduce_mean_weighted( hidden_r , skip_cell_weights_right , mean=True)
        pool_left_prim = reduce_mean_weighted( hidden_l , skip_cell_weights_left , mean=True)

        left_x_auxin = Concatenate(-1)([skip_cell_weights_left , left_x])
        right_x_auxin = Concatenate(-1)([skip_cell_weights_right , right_x])

        hidden_l_aux = rnn_left_aux( left_x_auxin )
        hidden_r_aux = rnn_right_aux( right_x_auxin )

        pool_right_aux = reduce_mean_weighted( hidden_r_aux , skip_cell_weights_right , mean=True)
        pool_left_aux = reduce_mean_weighted( hidden_l_aux , skip_cell_weights_left , mean=True)


        feats = Concatenate()([  pool_left_prim , pool_right_prim , pool_right_aux, pool_left_aux , pool_t ])
        feats_juss_aux = Concatenate()([   pool_right_aux, pool_left_aux  ])

        out = Dense(3 , activation='softmax' )( feats  )
        out_just_aux = Dense(3 , activation='softmax' )( feats_juss_aux  )


        rnn_left_aux_cell.set_weights(self.aux_w_l)
        rnn_right_aux_cell.set_weights(self.aux_w_r)


        return [left_i , right_i , tar_i ,sent_len_l ,sent_len_r , sent_len_t ] ,  [out,out_just_aux]



    def build_model(self):


        self.get_glove()

        self.aux_w_l , self.aux_w_r = self.get_document_level_rnn_weights()
        self.rnn_left_aux_cell__ = GRUCell(64 , trainable=False )
        self.rnn_right_aux_cell__ = GRUCell(64 , trainable=False )

        aux_inps , aux_outs = self.getAuxM()
        prim_inps , prim_outs = self.getPrimM()

        self.model = Model( prim_inps+aux_inps , prim_outs+aux_outs )
        Trainer.build_model( self  )
        
"""
        
config = {}
config['epochs'] = 4
config['dataset'] = "../../data/prepped/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
config['dataset'] =  "../../data/prepped/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
config['maxSentenceL'] = 80
config['maxTarLen'] = 10
config['nHidden'] = 64
config['dropout'] = 0.2
config['recurrent_dropout'] = 0.2

model = GIRNet_ABSA( exp_location="./ttt" , config_args = config )
model.train()

"""