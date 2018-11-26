# GIRNet: Interleaved Multi-Task Recurrent State Sequence Models
Code and datasets for our AAAI'19 paper : GIRNet: Interleaved Multi-Task Recurrent State Sequence Models.

The code is implemented in Keras

## Getting Started

### Prerequisites

* Keras 2.1.4
* Tensorflow 1.4.0
* Numpy 1.14.3
* Python 2.7
* h5py



```shell
pip install --upgrade keras==2.1.4
pip install --upgrade tensorflow-gpu==1.4.0
pip install --upgrade numpy==1.14.3
pip install h5py
pip install scikit-learn
```



### Using GIRNet

To use GIRNet, import GIRNet.py in your project. Refer the following snippet :

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



## Questions?

contact : divam14038 [at] iiitd [dot] ac [dot] in