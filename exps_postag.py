def free_tf_mem():
    import keras.backend.tensorflow_backend
    if keras.backend.tensorflow_backend._SESSION:
        import tensorflow as tf
        tf.reset_default_graph()
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None


def action(actset):
    from sequence_labeling import GIRNet_SeqLab, CrossStitch_SeqLab , HardShare_SeqLab , SharedPrivate_SeqLab , LowSup_SeqLab , Sluice_SeqLab
    if 0 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"
        config['exp_name'] = 'pos_girnet_1l'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        model = GIRNet_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 1 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"
        config['exp_name'] = 'pos_xstitch_1l'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['n_layers'] = 1 # 1 or 2 
        model = CrossStitch_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 2 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"
        config['exp_name'] = 'pos_xstitch_2l'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['n_layers'] = 2 # 1 or 2 
        model = CrossStitch_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 3 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"
        config['exp_name'] = 'pos_hardshare_1l'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['n_layers'] = 1
        model = HardShare_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 4 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_hardshare_2l'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['n_layers'] = 2
        model = HardShare_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 5 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_psp'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['mode'] = "parallel" # 'parallel' or 'stacked'
        model = SharedPrivate_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 6 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_ssp'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['mode'] = "stacked" # 'parallel' or 'stacked'
        model = SharedPrivate_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 7 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_lowsupconcat'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['mode'] = "concat" # concat or share
        model = LowSup_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 8 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_lowsupshare'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19
        config['mode'] = "share" # concat or share
        model = LowSup_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model

    if 9 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/postag_prepped.h5"

        config['exp_name'] = 'pos_sluice'
        config['embed_dim'] = 50
        config['vocab_size'] = 30003
        config['nHidden'] = 100
        config['sent_len'] = 150
        config['n_class_en'] = 45
        config['n_class_hi'] = 25
        config['n_class_enhi'] = 19

        model = Sluice_SeqLab( exp_location="outputs" , config_args = config )
        model.train()
        del model



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        action(range(10))
    else:
        action(map(lambda x: int(x), sys.argv[1:]))
