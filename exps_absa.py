def free_tf_mem():
    import keras.backend.tensorflow_backend
    if keras.backend.tensorflow_backend._SESSION:
        import tensorflow as tf
        tf.reset_default_graph()
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None


def action(actset):
    from target_based_classification import \
        CrossStitch_ABSA, SharedPrivate_ABSA, Sluice_ABSA, \
        LowSupShare_ABSA, HardShare_ABSA, GIRNet_ABSA

    if 0 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 2
        config['exp_name'] = 'xstitch_2l_laptop'
        free_tf_mem()
        model = CrossStitch_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 1 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 2
        config['exp_name'] = 'xstitch_2l_restaurants'
        free_tf_mem()
        model = CrossStitch_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 2 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'xstitch_1l_laptop'
        free_tf_mem()
        model = CrossStitch_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 3 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'xstitch_1l_restaurants'
        free_tf_mem()
        model = CrossStitch_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 4 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['sharing_scheme'] = 'parallel'  # 'parallel' or 'stacked'
        config['exp_name'] = 'psp_laptop'
        free_tf_mem()
        model = SharedPrivate_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 5 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['sharing_scheme'] = 'parallel'  # 'parallel' or 'stacked'
        config['exp_name'] = 'psp_restaurent'
        free_tf_mem()
        model = SharedPrivate_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 6 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['sharing_scheme'] = 'stacked'  # 'parallel' or 'stacked'
        config['exp_name'] = 'ssp_laptop'
        free_tf_mem()
        model = SharedPrivate_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 7 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['sharing_scheme'] = 'stacked'  # 'parallel' or 'stacked'
        config['exp_name'] = 'ssp_restaurent'
        free_tf_mem()
        model = SharedPrivate_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 8 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['exp_name'] = 'sluice_restaurent'
        free_tf_mem()
        model = Sluice_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 9 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['exp_name'] = 'sluice_laptop'
        free_tf_mem()
        model = Sluice_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 10 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['exp_name'] = 'lowsupshare_laptop'
        free_tf_mem()
        model = LowSupShare_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 11 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['exp_name'] = 'lowsupshare_restaurent'
        free_tf_mem()
        model = LowSupShare_ABSA(exp_location="outputs" , config_args=config)
        model.train()

    if 12 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'hardshare1l_restraunt'
        free_tf_mem()
        model = HardShare_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 13 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 2
        config['exp_name'] = 'hardshare2l_restraunt'
        free_tf_mem()
        model = HardShare_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 14 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'hardshare1l_laptop'
        free_tf_mem()
        model = HardShare_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 15 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 2
        config['exp_name'] = 'hardshare2l_laptop'
        free_tf_mem()
        model = HardShare_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 16 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'girnet_1l_restraunt'
        free_tf_mem()
        model = GIRNet_ABSA(exp_location="outputs", config_args=config)
        model.train()

    if 17 in actset:
        config = dict()
        config['epochs'] = 4
        config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
        config['maxSentenceL'] = 80
        config['maxTarLen'] = 10
        config['nHidden'] = 64
        config['dropout'] = 0.2
        config['recurrent_dropout'] = 0.2
        config['n_layers'] = 1
        config['exp_name'] = 'girnet_1l_laptop'
        free_tf_mem()
        model = GIRNet_ABSA(exp_location="outputs", config_args=config)
        model.train()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        action(range(18))
    else:
        action(map(lambda x: int(x), sys.argv[1:]))
