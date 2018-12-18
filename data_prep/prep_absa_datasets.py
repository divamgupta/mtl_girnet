import numpy as np
from dlblocks.pyutils import mapArrays , loadJson , saveJson , selectKeys , oneHotVec , padList
from dlblocks.pyutils import int64Arr , floatArr
import h5py

raw_absa_path = "../data/ABSC/data/absa/"
glove_h5_path = "../data/glovePrepped.h5"


# load the Glove Vectors 
gf = h5py.File(glove_h5_path)

import json
glove_name = "glove_common_42"
glovevocab = json.loads( gf['%s_vocab'%(glove_name)].value )
gloveVecs = np.array( gf['%s_vecs'%(glove_name)] )
def gloveDict( w ):
    if w in glovevocab:
        return glovevocab[w]
    else:
        return 0
    
    
    
print("glove vocal len" , len( glovevocab ))

def getrestraunt( split ):
    if split=='train':
        lines = open(raw_absa_path+  "restaurant/rest_2014_train.txt").read().split("\n")
    elif split == 'test':
        lines = open(raw_absa_path + "restaurant/rest_2014_test.txt").read().split("\n")
    lines = lines[:-1]
    lines = map( lambda x : x.strip() , lines )
    train = []
    sentences = lines [ 0:: 3]
    entities = lines[1::3]
    sentiments = lines[2::3]
    sentiments = map( int , sentiments )
    dset = zip( sentences , entities , sentiments )
    print(len( sentiments),len(entities) , len( sentences))
    assert len( sentiments)==len(entities) and len( entities)==len(sentences)
    dset = map( lambda x : {"sentence":x[0] , "entities":[{"entity": x[1] , "sentiment" : x[2] }]} , dset )
    return dset



def getlaptop( split ):
    if split=='train':
        lines = open(raw_absa_path + "laptop/laptop_2014_train.txt").read().split("\n")
    elif split == 'test':
        lines = open(raw_absa_path+  "laptop/laptop_2014_test.txt").read().split("\n")
    lines = lines[:-1]
    lines = map( lambda x : x.strip() , lines )
    train = []
    sentences = lines [ 0:: 3]
    entities = lines[1::3]
    sentiments = lines[2::3]
    sentiments = map( int , sentiments )
    dset = zip( sentences , entities , sentiments )
    print(len( sentiments),len(entities) , len( sentences))
    assert len( sentiments)==len(entities) and len( entities)==len(sentences)
    dset = map( lambda x : {"sentence":x[0] , "entities":[{"entity": x[1] , "sentiment" : x[2] }]} , dset )
    return dset





def vecc( d ):
    ret = {}
        
    entityWords = d['entities'][0]['entity'].lower().strip().split()
    sentiment = d['entities'][0]['sentiment']

    leftWords = d['sentence'].split("$T$")[0].lower().strip().split()
    rightWords = d['sentence'].split("$T$")[1].lower().strip().split()
    
    words = leftWords + entityWords + rightWords
    entityMask = [0]*len( leftWords) + [1]*len(entityWords) + [0]*(len(rightWords))

    wordids = map( gloveDict , words )
    wordIdsLeft = map( gloveDict , leftWords )
    wordIdsRight = map( gloveDict , rightWords )
    wordIdsEntities = map( gloveDict , entityWords )

    
    ret['sentence_glove_rightpad'] = int64Arr( padList( wordids , maxSentenceL , 0 , 'right') )
    ret['sentence_left_glove_rightpad']  = int64Arr( padList( wordIdsLeft , maxSentenceL , 0 , 'right') )
    ret['sentence_right_glove_rightpad']   = int64Arr( padList( wordIdsRight[::-1] , maxSentenceL , 0 , 'right') )
    
    ret['sentence_right_len'] = len( wordIdsRight ) 
    ret['sentence_left_len'] = len( wordIdsLeft ) 
    
    ret['sentence_entity_glove_rightpad']   = int64Arr( padList( wordIdsEntities , 20  , 0 , 'right') )
    ret['sentence_entity_len']   = len( wordIdsEntities )


    ret['sentiment_onehot'] =  floatArr( oneHotVec( sentiment +1 , 3  ) )

    return ret



for dfn , outN in [ (getlaptop , "semival14_absa_Laptop_prepped_V2_gloved_42B" ) , ( getrestraunt , "semival14_absa_Restaurants_prepped_V2_gloved_42B") ] :
    outFNN = "../data/%s.h5"%outN
    print(outFNN)
    
    train = dfn(split = "train"    )
    test = dfn(split = "test"   )
    
    maxSentenceL = 160

    train_arr = mapArrays( train , vecc )
    test_arr = mapArrays( test , vecc )

    f = h5py.File(outFNN , "w")
    f.create_group("train")
    for k in train_arr.keys():
        f['train'].create_dataset( k , data=train_arr[k ])
    f.create_group("test")
    for k in test_arr.keys():
        f['test'].create_dataset( k , data=test_arr[k ])
    f.close()




