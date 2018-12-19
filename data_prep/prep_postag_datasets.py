"""
Data source :

# https://github.com/ltrc/hin-POS-tagger/tree/master/data
# https://www.clips.uantwerpen.be/conll2000/chunking/
# http://www.amitavadas.com/Code-Mixing.html

"""


import json
import h5py
import numpy as np
import glob
import os
import random
from keras.utils import to_categorical
from dlblocks import text
from dlblocks.pyutils import mapArrays , loadJson , saveJson , selectKeys , oneHotVec , padList
from dlblocks.pyutils import int64Arr , floatArr
from dlblocks.pyutils import loadJson , selectKeys
import numpy as np
import json


raw_data_path = "./data_raw_postag/"


hi_en_words = loadJson(raw_data_path + "higlish2hindi-simple.json")
hi2en = {}
en2hi = {}

for d in hi_en_words:
    hi2en[ d['hindi'] ] = d['hinglish']
    en2hi[ d['hinglish'] ] = d['hindi']
    
print "loaded the devanagari -> roman mappings"


print 'eg : shayad ->' , en2hi['shayad']



# Prep the english aux dataset

data = open( raw_data_path +  "en_train.txt").read()
data = data.split("\n\n")
data = map( lambda x: x.split("\n") , data)[:-1]
allPostags = []

for i,d in enumerate(data):
    d = map( lambda x:x.split(' ') , d)
    d = map( lambda x: {'word':x[0].lower(), 'word_main':x[0], 'pos':x[1]} , d)
    allPostags += selectKeys(d , 'pos')
    d[0]['postag_set'] = 'postags_en'
    data[i] = d
allPostags = list(set(allPostags))
data_en = data
postags_en = allPostags
print "English Aux dataset prepped"




# Prep the hindi aux dataset

data = open( raw_data_path + "training_wx.txt").read().split("</Sentence>")[:-1]

allPostags = []

for i,d in enumerate(data):
    d = d.split("\n")
    d = filter( lambda x: "name='"in x and "<fs af=" in x and ">"in x , d )
    posTags = map( lambda x:x.split('\t')[2], d)
    words = map( lambda x:x.split("name='")[1].split("'")[0], d)
    words = map( lambda x : "".join([ c for c in list(x) if c not in map(str,range(9) )  ]) , words)
    words = map( lambda x: json.loads(json.dumps(x)) , words)
    d = map( lambda x: {'word_hi':x[0], 'word_main':x[0], 'pos':x[1]} , zip( words , posTags ))
    d[0]['postag_set'] = 'postags_hi'
    data[i] = d
    allPostags += selectKeys(d , 'pos')
allPostags = list(set(allPostags))

data_hi = data
postags_hi = allPostags
print "Hindi Aux dataset prepped"




# Prep the code-mix primary dataset

d1 = open( raw_data_path + "FB_HI_EN_CR.txt").read().split("\n\n")[:-1]
d2 = open( raw_data_path + "TWT_HI_EN_CR.txt").read().split("\n\n")[:-1]
d3 = open( raw_data_path + "WA_HI_EN_CR.txt").read().split("\n\n")[:-1]

data_enhi_1_tr = []
data_enhi_1_te = []

allPostags = []

for data in [d1 , d2 , d3 ]:

    for i,d in enumerate(data):

        d = d.split("\n")
        d = map( lambda x:x.split("\t") , d)

        words = map( lambda x:x[0] , d )
        words = [ w.lower() for w in words]


        words_hi = ['<unk>']*len(words)
        langs = map( lambda x:x[1] , d )
        posTags = map( lambda x:x[2] , d )
        d = map( lambda x: {'word':x[0],'pos':x[1] , 'word_hi':x[2]} , zip( words , posTags , words_hi))



        for j in range(len(d)):
            if d[j]['word'] in en2hi:
                 d[j]['word_hi'] = en2hi[d[j]['word'] ]
            d[j]['word_hi']  = json.loads(json.dumps(d[j]['word_hi']))

            if langs[j] == 'hi':
                d[j]['word_main'] = d[j]['word_hi'] 
            else:
                d[j]['word_main'] = d[j]['word'] 

        allPostags += selectKeys(d , 'pos')
        d[0]['postag_set'] = 'postags_enhi_1'
        data[i] = d

    allPostags = list(set(allPostags))
    data_enhi_1 = data
    postags_enhi_1 = allPostags

    data_enhi_1_tr += data_enhi_1[: int(0.8*len(data_enhi_1))]
    data_enhi_1_te += data_enhi_1[ int(0.8*len(data_enhi_1)) : ]

print "Code-mix prim dataset prepped"


vocab_hi = text.Vocabulary()
vocab_en = text.Vocabulary()
vocab_main = text.Vocabulary()


for d in data_hi :
    vocab_hi.add_words( selectKeys(d , 'word_hi' )  )
    
for d in data_enhi_1 :
    vocab_hi.add_words( selectKeys(d , 'word_hi' )  )
    
for d in data_en :
    vocab_en.add_words( selectKeys(d , 'word' )  )
    
for d in data_enhi_1 :
    vocab_en.add_words( selectKeys(d , 'word' )  )
    
    
vocab_en.keepTopK(15000)
vocab_hi.keepTopK(15000)




for wrd in vocab_hi.word2idx.keys()+vocab_en.word2idx.keys():
    vocab_main.add_word( wrd )
    
    
print "vocab generated "


maxSentenceL = 150
def vecc( d ):
    ret = {}
    
    pos_tag_set = eval(d[0]['postag_set'])
    
    words = selectKeys(d , 'word_main')
    wordids = map( vocab_main , words )
    
    ret['sentence_main'] = int64Arr( padList( wordids , maxSentenceL , 0 , 'right') )
        
    posTags = selectKeys(d , 'pos')
    posTagIds = list(np.array([ pos_tag_set.index( p ) for p in  posTags  ]).astype(int)+1)
    posTagIds = np.array( padList( posTagIds , maxSentenceL , 0 , 'right') ).astype(np.uint8)
    posTagIds_oh = to_categorical( np.array(posTagIds) , len(pos_tag_set)+1 )
    ret['posTagIds_oh'] = posTagIds_oh
    return ret



datasets = { }

datasets['data_en'] = mapArrays( data_en , vecc )
datasets['data_hi'] = mapArrays( data_hi , vecc )
datasets['data_enhi_1_tr'] = mapArrays( data_enhi_1_tr , vecc )
datasets['data_enhi_1_te'] = mapArrays( data_enhi_1_te , vecc )

outFNN = "../data/postag_prepped.h5"

f = h5py.File(outFNN , "w")
for kk in datasets.keys():
    f.create_group( kk  )
    for k in datasets[kk].keys():
        f[ kk ].create_dataset( k , data=datasets[kk][k] )

print "HDF5 file created !"


