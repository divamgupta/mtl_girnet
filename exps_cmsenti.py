def free_tf_mem():
    import keras.backend.tensorflow_backend
    if keras.backend.tensorflow_backend._SESSION:
        import tensorflow as tf
        tf.reset_default_graph()
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None


def action(actset):
    from sequence_classification import GIRNet_SeqClass, HardShare_SeqClass , LowSup_SeqClass , Sluice_SeqClass , SharedPrivate_SeqClass , CrossStitch_SeqClass
    
    if 0 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_girnet_1l'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3

        model = GIRNet_SeqClass( exp_location="outputs" , config_args = config )
        model.train()


        del model

    if 1 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_harsdhare_1l'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['n_layers'] = 1

        model = HardShare_SeqClass( exp_location="outputs" , config_args = config )
        model.train()

        del model

    if 2 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_harsdhare_2l'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['n_layers'] = 2

        model = HardShare_SeqClass( exp_location="outputs" , config_args = config )
        model.train()

        del model

    if 3 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_lowsupconcat'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['mode'] = "concat" # share , concat

        model = LowSup_SeqClass( exp_location="outputs" , config_args = config )
        model.train()
        del model


    if 4 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_lowsupshare'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['mode'] = "share" # share , concat

        model = LowSup_SeqClass( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 5 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_psp'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['mode'] = "parallel" # stacked , parallel

        model = SharedPrivate_SeqClass( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 6 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_ssp'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['mode'] = "stacked" # stacked , parallel

        model = SharedPrivate_SeqClass( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 7 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_xstitch1l'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['n_layers'] = 1 # 1 , 2 

        model = CrossStitch_SeqClass( exp_location="outputs" , config_args = config )
        model.train()

        del model

    if 8 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_xstitch2l'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 64
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3
        config['n_layers'] = 2 # 1 , 2 

        model = CrossStitch_SeqClass( exp_location="outputs" , config_args = config )
        model.train()


    if 9 in actset:
        config = dict()
        config['epochs'] = 2
        config['dataset'] = "data/senti_prepped.h5"

        config['exp_name'] = 'cm_senti_sluice'
        config['embed_dim'] = 300
        config['vocab_size'] = 35000
        config['nHidden'] = 100
        config['sent_len'] = 50
        config['n_class_en'] = 3
        config['n_class_es'] = 3
        config['n_class_enes'] = 3

        model = Sluice_SeqClass( exp_location="outputs" , config_args = config )
        model.train()
        del model



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        action(range(10))
    else:
        action(map(lambda x: int(x), sys.argv[1:]))
