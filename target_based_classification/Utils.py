
from keras.models import *
from keras.layers import *

import keras
import keras.backend as K
from keras import initializers

import h5py

from sklearn.metrics import classification_report , accuracy_score , f1_score , recall_score , precision_score , mean_absolute_error , mean_squared_error

from myutils.keras_utils import WeightsPredictionsSaver
from myutils.TrainUtils import BaseTrainer
import random
import json
import os
import numpy as np



import numpy as np
import pickle




#from keras.utils import plot_model


def get_laststep_lstm( n_units ):
    return LSTM(   units = n_units  ,
                     return_sequences=False ,
                     kernel_initializer = initializers.glorot_uniform(seed=0) ,
                     recurrent_initializer = initializers.Orthogonal(seed=0)  )



def get_lstm( n_units ):
    return LSTM(   units = n_units  ,
                     return_sequences=True ,
                     kernel_initializer = initializers.glorot_uniform(seed=0) ,
                     recurrent_initializer = initializers.Orthogonal(seed=0)  )

def get_fc_classifier( n_units ):
    return Dense( units = n_units ,
                  activation='softmax' ,
                  kernel_initializer = initializers.glorot_uniform(seed=0) )


def get_embed(n_vocab=None , n_units=None , glove=False ):

    if not glove:
        return Embedding( n_vocab ,
                        n_units ,
                        embeddings_initializer = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=0 )
        )

    else:

        gf = h5py.File("../../data/prepped/glovePrepped.h5")
        gloveVecs_42 = np.array( gf['glove_common_42_vecs'] )

        gloveSize  = gloveVecs_42.shape[-1]
        vocabSize = gloveVecs_42.shape[0]

        glove_embed42 = (Embedding( vocabSize ,
                                   gloveSize ,
                                   weights=[gloveVecs_42] ,
                                   trainable=False ,
                                   embeddings_initializer = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=0 )))

        return glove_embed42


def eval_classification( gt_inds , pr_inds ):
    print "Accuracy " , accuracy_score(  gt_inds , pr_inds )
    print "F1 " , f1_score(  gt_inds , pr_inds, average='macro'  )
    print "Precision " , precision_score(  gt_inds , pr_inds , average='macro' )
    print "Recall " , recall_score(  gt_inds , pr_inds , average='macro' )







def eval_class( GT , PR ):
    d = {}
    #print classification_report( GT , PR , digits=5)
    d[ "accuracy" ] =   accuracy_score( GT , PR )*100.0
    d[ "f1" ] = f1_score( GT , PR , average='micro'   )*100.0
    d[  "recall" ] = recall_score( GT , PR   , average='micro')*100.0
    d[ "precision "] =  precision_score( GT , PR  , average='micro' )*100.0
    d[ "macro f1" ] = f1_score( GT , PR , average='macro' )*100.0
    d[  "macro recall" ] = recall_score( GT , PR , average='macro' )*100.0
    d[ "macro precision " ] =  precision_score( GT , PR , average='macro' )*100.0

    for k in d:
        d[k] = round( float(d[k]) ,  2)

    return d




class Trainer( BaseTrainer ):
    """docstring for Trainer"""

    def __init__(self, **kargs ):
        BaseTrainer.__init__( self,  **kargs  )


    def set_dataset( self ):

        dataset_path = self.config['dataset']
        f = h5py.File( dataset_path , "r")
        f2 = h5py.File( "./data/yelp2014_glove42b.h5" , "r")
        
        maxSentenceL = self.config['maxSentenceL']
        maxTarLen = self.config['maxTarLen']

        dupp = lambda x:  np.concatenate([x]*3 )
        len_test =  len(f['test']['sentence_entity_len'])
        len_train = len( f['train']['sentence_entity_len'])


        tr_x = [ dupp( f['train']['sentence_left_glove_rightpad'][:  , : maxSentenceL   ] )
                ,  dupp( f['train']['sentence_right_glove_rightpad'][:  , : maxSentenceL  ] )
                ,  dupp( f['train']['sentence_entity_glove_rightpad'][:  , : maxTarLen  ] )
                ,  dupp( f['train']['sentence_left_len'] )
                ,  dupp( f['train']['sentence_right_len'] )
                ,  dupp( f['train']['sentence_entity_len'] )
                , f2['train']['sentence_glove_rightpad'][:3*len_train][:  , : maxSentenceL   ]
               ]

        tr_y =[ dupp(f['train']['sentiment_onehot']),  dupp(f['train']['sentiment_onehot'])
              , f2['train']['sentiment_onehot'][: 3*len_train ]  ]


        te_x = [ np.array(f['test']['sentence_left_glove_rightpad'][:  , : maxSentenceL   ] )
                ,  np.array(f['test']['sentence_right_glove_rightpad'][:  , : maxSentenceL  ] )
                ,  np.array(f['test']['sentence_entity_glove_rightpad'][:  , : maxTarLen  ] )
                ,  np.array(f['test']['sentence_left_len'] )
                ,  np.array(f['test']['sentence_right_len'] )
                ,  np.array( f['test']['sentence_entity_len'] )
                ,  np.array(f2['test']['sentence_glove_rightpad'][:len_test][:  , : maxSentenceL   ])
               ]


        te_y = [  np.array(  f['test']['sentiment_onehot'])
                , np.array(  f['test']['sentiment_onehot'])
                , np.array(  f2['test']['sentiment_onehot'][: len_test ] ) ]


        self.data_inp = tr_x
        self.data_target = tr_y
        self.te_data_inp = te_x
        self.te_data_data_target = te_y

        BaseTrainer.set_dataset( self  )


    def build_model(self):
        self.model.compile('adam' , 'categorical_crossentropy' , metrics=['accuracy'])
        BaseTrainer.build_model( self  )



    def evaluate( self ):

        d = pickle.load( open( os.path.join(  self.exp_location,self.exp_name  ) +".preds.pkl") )
        gt = d['GT'][0].argmax(-1)
        mpr = None
        mac = -1
        for i in range( len( d['PR']) ):
            if type(d['PR'][i]) is list:
                pr = d['PR'][i][0]
            else:
                pr = d['PR'][i]
            pr = pr.argmax(-1)

            acc = accuracy_score( gt , pr )

            if acc > mac:
                mac = acc
                mpr = pr

        d = eval_class( gt , mpr )
        d['exp_name'] = self.exp_name
        s = "expname:results " + self.exp_name+":("
        for k in d:
            s += str(k) +":"+ str(d[k])+"|"
        s += ")"
        print s
        return d



