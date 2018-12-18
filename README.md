# GIRNet: Interleaved Multi-Task Recurrent State Sequence Models

Packaged datasets and Keras code for the paper [GIRNet: Interleaved Multi-Task Recurrent State Sequence Models](https://arxiv.org/abs/1811.11456).

Prepare a virtual environment and install requirements as follows.
```shell
$ virtualenv -p `which python2` /path/to/girnet-env
$ source /path/to/girnet-env/bin/activate
(girnet-env)$ pip install -r requirements.txt
```

We will assume this code has been cloned to `/path/to/mtl_girnet` as the code base directory.  Download the [zipped data files](https://drive.google.com/open?id=1fksInwJMD9vlFfduonjDyNMJ5GbUkKTQ) and unzip in the code base directory, which will place all the .h5 files in the data subdirectory.  [Gdrive](https://github.com/prasmussen/gdrive) can be used for downloading.
```bash
$ cd /tmp
$ gdrive download 1fksInwJMD9vlFfduonjDyNMJ5GbUkKTQ
$ cd /path/to/mtl_girnet
$ unzip /path/to/zipfile
```
If you want to prepare the data sets by yourself, clone this [repository of labeled aspect-based sentiment data](https://github.com/NUSTM/ABSC.git) and convert to the .h5 format we use, by running
```bash
(girnet-env)$ cd /path/to/mtl_girnet/data
(girnet-env)$ git clone https://github.com/NUSTM/ABSC.git
(girnet-env)$ cd /path/to/mtl_girnet/data_prep
(girnet-env)$ python prep_absa_datasets.py
```

To use GIRNet, import GIRNet.py in your project. Examples are provided in the following snippet.
```python
# Import GIRNet
from GIRNet import GIRNet

# Define the aux LSTMs
rnn_aux1 = LSTM( nHidden )
rnn_aux2 = LSTM( nHidden )

# Submodel for aux task 1
inp_aux1 = Input((n ,))
x_a1 = rnn_aux1( inp_aux1 )
out_aux1 = Dense( n_classes , activation='softmax')( x_a1 )

# Submodel for aux task 2
inp_aux2 = Input((n ,))
x_a2 = rnn_aux1( inp_aux2 )
out_aux2 = Dense( n_classes , activation='softmax')( x_a2 )

# Submodel for prim task
inp_prim = Input((n,))
gate_vales , prime_out , out_interleaved = GIRNet( inp_prim ,  [rnn_aux1 , rnn_aux2 ] , return_sequences=False )
out_prim = Dense( 3 , activation='softmax')( out_interleaved )

m = Model([inp_aux1 , inp_aux2 , inp_prim] , [out_aux1  , out_aux2  , out_prim ] )
```

In case of questions, contact: divam14038 [at] iiitd [dot] ac [dot] in
