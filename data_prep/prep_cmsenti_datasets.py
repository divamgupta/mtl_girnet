import json
import h5py
import numpy as np
import glob
import os
import random


from twtokenize import tokenize
from keras.utils import to_categorical
from xml.dom import minidom

from dlblocks import text
from dlblocks.pyutils import mapArrays , loadJson , saveJson , selectKeys , oneHotVec , padList
from dlblocks.pyutils import int64Arr , floatArr




sents = {"N":-1 , "P" :1 , "NONE":0}


data = open("./data_cm_senti/cs-corpus-with-tweets_train.txt").read().split("\n") 
data = map( lambda x : x.split("\t") , data )
data = map( lambda x :{'sentiment': sents[x[1]] , 'tokens': tokenize(x[2]) , 'text': x[2] } , data )
en_es_wssa_data_train = data


data = open("./data_cm_senti/cs-corpus-with-tweets_test.txt").read().split("\n") 
data = map( lambda x : x.split("\t") , data )
data = map( lambda x :{'sentiment': sents[x[1]] , 'tokens': tokenize(x[2]) , 'text': x[2] } , data )
en_es_wssa_data_test = data

en_es_wssa_data = en_es_wssa_data_train + en_es_wssa_data_test





xmldoc = minidom.parse("./data_cm_senti/general-tweets-train-tagged.xml")
tweets = xmldoc.getElementsByTagName('tweet')

sents = {"N":-1 , "P" :1 , "NEU":0 , 'NONE':0 , "P+" : 1 , "N+":-1 }


es_tass1_data = []

for i in range( len(tweets)-1) :
    if i == 6055:
        continue # bad jogar
    textt = tweets[i].getElementsByTagName('content')[0].childNodes[0].data
    words = tokenize( textt )
    sentiment = tweets[i].getElementsByTagName('polarity')[0].getElementsByTagName('value')[0].childNodes[0].data
    assert len(tweets[i].getElementsByTagName('polarity')[0].getElementsByTagName('entity'))==0
    es_tass1_data.append({'text':textt , 'tokens':words , 'sentiment': sents[sentiment] })
    


data = open("./data_cm_senti/twitter4242.txt").read().split("\n")[1:-1]
data = map( lambda x : x.split("\t") , data )
data = map( lambda x :{'sentiment': int(np.sign(int(x[0])-int(x[1]))) , 'tokens': tokenize(x[2]) , 'text': x[2] } , data )

en_twitter_data = data






data = open("./data_cm_senti/1600_tweets_dev_complete.txt").read().split("\n")[1:-1]
data += open("./data_cm_senti/1600_tweets_test_average_complete.tsv").read().split("\n")[1:-2]

data = map( lambda x : x.split("\t") , data )
data = map( lambda x :{'sentiment': int(np.sign(int(x[0])-int(x[1]))) , 'tokens': tokenize(x[2]) , 'text': x[2] } , data )

es2_twitter_data = data







vocab = text.Vocabulary()

for d in es_tass1_data + en_es_wssa_data + en_twitter_data + es2_twitter_data :
    vocab.add_words( d['tokens']  )

    
vocab.keepTopK(25000)



maxSentenceL = 150

def vecc( d ):
    ret = {}
    words   = d['tokens']
    wordids = map( vocab , words )
    ret['sentence'] = int64Arr( padList( wordids , maxSentenceL , 0 , 'left') )
    ret['sentiment_val'] =  floatArr( d['sentiment'] )
    ret['sentiment_id'] =  int64Arr( d['sentiment'] + 1 )
    ret['sentiment_onehot'] =  floatArr( oneHotVec( d['sentiment']+1 , 3  ) )

    return ret





en_es_wssa_data_train_arr = mapArrays( en_es_wssa_data_train , vecc )
en_es_wssa_data_test_arr = mapArrays( en_es_wssa_data_test , vecc )

en_twitter_data_train_arr = mapArrays( en_twitter_data , vecc )
es_tass1_datatrain_arr = mapArrays( es_tass1_data , vecc )

datasets = {"en_es_wssa_data_train_arr":en_es_wssa_data_train_arr ,
           "en_es_wssa_data_test_arr":en_es_wssa_data_test_arr ,
           "en_twitter_data_train_arr":en_twitter_data_train_arr ,
           "es_tass1_datatrain_arr": es_tass1_datatrain_arr }



outFNN = "../data/senti_prepped.h5"

f = h5py.File(outFNN , "w")
for kk in datasets.keys():
    f.create_group( kk  )
    for k in datasets[kk].keys():
        f[ kk ].create_dataset( k , data=datasets[kk][k] )

print "HDF5 file created !"








