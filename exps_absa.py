

from target_based_classification import *
import sys



en = int(  sys.argv[1] )

if en == 0:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 2
	config['exp_name'] = 'xstitch_2l_laptop'
	model = CrossStitch_ABSA( exp_location="outputs" , config_args = config )
	model.train()



if en == 1:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 2
	config['exp_name'] = 'xstitch_2l_restaurants'
	model = CrossStitch_ABSA( exp_location="outputs" , config_args = config )
	model.train()




if en == 2:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'xstitch_1l_laptop'
	model = CrossStitch_ABSA( exp_location="outputs" , config_args = config )
	model.train()




if en == 3:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'xstitch_1l_restaurants'
	model = CrossStitch_ABSA( exp_location="outputs" , config_args = config )
	model.train()









if en == 4:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['sharing_scheme'] = 'parallel'  # 'parallel' or 'stacked'
	config['exp_name'] = 'psp_laptop'
	model = SharedPrivate_ABSA( exp_location="outputs" , config_args = config )
	model.train()



if en == 5:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['sharing_scheme'] = 'parallel'  # 'parallel' or 'stacked'
	config['exp_name'] = 'psp_restaurent'
	model = SharedPrivate_ABSA( exp_location="outputs" , config_args = config )
	model.train()






if en == 6:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['sharing_scheme'] = 'stacked'  # 'parallel' or 'stacked'
	config['exp_name'] = 'ssp_laptop'
	model = SharedPrivate_ABSA( exp_location="outputs" , config_args = config )
	model.train()






if en == 7:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['sharing_scheme'] = 'stacked'  # 'parallel' or 'stacked'
	config['exp_name'] = 'ssp_restaurent'
	model = SharedPrivate_ABSA( exp_location="outputs" , config_args = config )
	model.train()


if en == 8:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['exp_name'] = 'sluice_restaurent'
	model = Sluice_ABSA( exp_location="outputs" , config_args = config )
	model.train()

if en == 9:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['exp_name'] = 'sluice_laptop'
	model = Sluice_ABSA( exp_location="outputs" , config_args = config )
	model.train()



if en == 10:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['exp_name'] = 'lowsupshare_laptop'
	model = LowSupShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()


if en == 11:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['exp_name'] = 'lowsupshare_restaurent'
	model = LowSupShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()



if en == 12:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'hardshare1l_restraunt'
	model = HardShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()


if en == 13:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 2
	config['exp_name'] = 'hardshare2l_restraunt'
	model = HardShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()



if en == 14:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'hardshare1l_laptop'
	model = HardShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()


if en == 15:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 2
	config['exp_name'] = 'hardshare2l_laptop'
	model = HardShare_ABSA( exp_location="outputs" , config_args = config )
	model.train()






if en == 16:
	config = {}
	config['epochs'] = 4
	config['dataset'] =  "data/semival14_absa_Restaurants_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'girnet_1l_restraunt'
	model = GIRNet_ABSA( exp_location="outputs" , config_args = config )
	model.train()


if en == 17:
	config = {}
	config['epochs'] = 4
	config['dataset'] = "data/semival14_absa_Laptop_prepped_V2_gloved_42B.h5"
	config['maxSentenceL'] = 80
	config['maxTarLen'] = 10
	config['nHidden'] = 64
	config['dropout'] = 0.2
	config['recurrent_dropout'] = 0.2
	config['n_layers'] = 1
	config['exp_name'] = 'girnet_1l_laptop'
	model = GIRNet_ABSA( exp_location="outputs" , config_args = config )
	model.train()






