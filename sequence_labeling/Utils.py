
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
        
        dup = lambda x: np.concatenate([x]*(int(15000/x.shape[0])+1)  )
        
        
        datasets = {}
        for kk in f.keys():
            datasets[ kk ] = {}
            for k in f[kk].keys():
                datasets[kk][k] = np.array( f[ kk ][k] )
                
        n_test = (datasets['data_enhi_1_te']['sentence_main']).shape[0]
        
        
        n_p = 15000
        
                
        x_tr = [
            dup(datasets['data_hi']['sentence_main'])[:n_p ] ,
            dup(datasets['data_en']['sentence_main'])[:n_p ] , 
            dup(datasets['data_enhi_1_tr']['sentence_main'])[:n_p ] ,
        ]

        x_te = [
            dup(datasets['data_hi']['sentence_main'])[:n_test ] ,
            dup(datasets['data_en']['sentence_main'])[:n_test ] , 
            dup(datasets['data_enhi_1_te']['sentence_main'])[:n_test ] ,
        ]

        y_tr = [
            dup(datasets['data_hi']['posTagIds_oh'])[:n_p ] ,
            dup(datasets['data_en']['posTagIds_oh'])[:n_p ] , 
            dup(datasets['data_enhi_1_tr']['posTagIds_oh'])[:n_p ] 
        ]

        y_te = [
            (datasets['data_hi']['posTagIds_oh'])[:n_test ] ,
            (datasets['data_en']['posTagIds_oh'])[:n_test ] , 
            (datasets['data_enhi_1_te']['posTagIds_oh'])[:n_test ] 
        ]

        
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
        
        mpr = None
        mac = -1
        for i in range( len( d['PR']) ):
            if len( d['GT'][-1].shape) == 2:
                gt = d['GT'][-1]
            else:
                gt = d['GT'][-1].argmax(-1)
            if type(d['PR'][i]) is list:
                pr = d['PR'][i][-1]
            else:
                pr = d['PR'][i]
            pr = pr.argmax(-1)
            pr = pr[gt > 0]
            gt = gt[gt > 0]

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

