
from keras.models import *
from keras.layers import *

import keras
import keras.backend as K
from keras import initializers

import h5py

from sklearn.metrics import classification_report , accuracy_score , f1_score , recall_score , precision_score , mean_absolute_error , mean_squared_error

from dlblocks.keras_utils import WeightsPredictionsSaver
from dlblocks.TrainUtils import BaseTrainer
import random
import json
import os
import numpy as np
import pickle


def eval_class( GT , PR ):
    d = {}
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
        
        dupp = lambda x:  np.concatenate([x]*5)
                
        datasets = {}
        for kk in f.keys():
            datasets[ kk ] = {}
            for k in f[kk].keys():
                datasets[kk][k] = np.array( f[ kk ][k] )
                
                
        en_es_wssa_data_train_arr = datasets['en_es_wssa_data_train_arr']
        en_es_wssa_data_test_arr = datasets['en_es_wssa_data_test_arr']
        en_twitter_data_train_arr = datasets['en_twitter_data_train_arr']
        es_tass1_datatrain_arr = datasets['es_tass1_datatrain_arr']
        
        
        x_enes = dupp(en_es_wssa_data_train_arr['sentence'][: , -50:])
        y_enes = dupp(en_es_wssa_data_train_arr['sentiment_onehot'])

        x_enes_te = en_es_wssa_data_test_arr['sentence'][: , -50:]
        y_enes_te = en_es_wssa_data_test_arr['sentiment_onehot']

        n_test = x_enes_te.shape[0]

        x_en = dupp(en_twitter_data_train_arr['sentence'][: , -50:])
        y_en = dupp(en_twitter_data_train_arr['sentiment_onehot'])

        x_es = dupp(es_tass1_datatrain_arr['sentence'][: , -50:])
        y_es = dupp(es_tass1_datatrain_arr['sentiment_onehot'])

        x_tr = [x_en[:3062]  ,  x_es[:3062] , x_enes[:3062] ]
        y_tr = [y_en[:3062]  ,  y_es[:3062] , y_enes[:3062] ]


        x_te = [x_en[:n_test]  ,  x_es[:n_test] , x_enes_te  ]
        y_te = [y_en[:n_test]  ,  y_es[:n_test] , y_enes_te  ]

        
        self.data_inp = x_tr
        self.data_target = y_tr
        self.te_data_inp = x_te
        self.te_data_data_target = y_te

        BaseTrainer.set_dataset( self  )


    def build_model(self):
        self.model.compile('adam' , 'categorical_crossentropy' , metrics=['accuracy'])
        BaseTrainer.build_model( self  )
        
        
    def evaluate(self ):

        d = pickle.load( open( os.path.join(  self.exp_location,self.exp_name  ) +".preds.pkl") )


        if len( d['GT'][-1].shape) == 1:
            gt = d['GT'][-1]
        else:
            gt = d['GT'][-1].argmax(-1)
        mpr = None
        mac = -1
        for i in range( len( d['PR']) ):
            if type(d['PR'][i]) is list:
                pr = d['PR'][i][-1]
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


    
