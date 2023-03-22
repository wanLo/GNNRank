import os

import numpy as np
import torch
from texttable import Texttable

from src.param_parser import parameter_parser
from src.metrics import print_performance_mean_std
from src.Trainer import Trainer

GNN_variant_names = ['dist', 'innerproduct', 'proximal_dist', 'proximal_innerproduct', 'proximal_baseline']
NUM_GNN_VARIANTS = len(GNN_variant_names) # number of GNN variants for each architecture

upset_choices = ['upset_simple', 'upset_ratio', 'upset_naive']
NUM_UPSET_CHOICES = len(upset_choices)
args = parameter_parser()
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
compare_names_all = []
for method_name in args.all_methods:
    if method_name not in ['DIGRAC', 'ib']:
        compare_names_all.append(method_name)
    else:
        for GNN_type in GNN_variant_names:
            compare_names_all.append(method_name+'_'+GNN_type)



### this is where the class definition used to be ###


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/'+args.dataset)


kendalltau_res = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 2])
kendalltau_res_latest = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 2])

final_upset = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), NUM_UPSET_CHOICES])
final_upset[:] = np.nan
final_upset_latest = final_upset.copy()

method_str = ''
for method_name in args.all_methods:
    method_str += method_name

default_name_base = ''
if 'DIGRAC' in args.all_methods or 'ib' in args.all_methods:
    default_name_base += 'K' + str(args.K) + 'dropout' + str(int(100*args.dropout))
    default_name_base += 'ratio_coe' + str(int(100*args.upset_ratio_coeff)) + 'margin_coe' + str(int(100*args.upset_margin_coeff)) 
    if args.upset_margin_coeff > 0:
        default_name_base += 'margin' + str(int(100*args.upset_margin)) 
    default_name_base += 'with' + str(args.train_with)  + 'Fiedler' + str(args.Fiedler_layer_num) + 'sigma' + str(int(100*args.sigma))
    default_name_base += 'alpha' + str(int(100*args.alpha))
    if args.train_with[:8] == 'proximal':
        default_name_base += 'train_alpha' + str(args.trainable_alpha)
    default_name_base += 'hid' + str(args.hidden) + 'lr' + str(int(1000*args.lr))
    default_name_base += 'use' + str(args.baseline)
    if args.pretrain_epochs > 0 and args.train_with[:8] == 'proximal':
        default_name_base +=  'pre' + str(args.pretrain_with) + str(int(args.pretrain_epochs))
save_name_base = default_name_base

default_name_base +=  'trials' + str(args.num_trials) + 'train_r' + str(int(100*args.train_ratio)) + 'test_r' + str(int(100*args.test_ratio)) + 'All' + str(args.AllTrain)
save_name_base = default_name_base
if args.dataset[:3] == 'ERO':
    default_name_base += 'seeds' + '_'.join([str(value) for value in np.array(args.seeds).flatten()])
save_name = default_name_base


### the actuall training follows ###


current_seed_ind = 0
for random_seed in args.seeds:
    current_ind = 0
    trainer = Trainer(args, random_seed, save_name_base)
    for method_name in args.all_methods:

        if method_name in ['DIGRAC', 'ib']:

            ### train the NN models
            save_path_best, save_path_latest = trainer.train(model_name=method_name)
            
            ### evaluate best model
            print('\n--- best model ---')
            kendalltau_full = []
            upset_full = []
            # evaluate for all GNN variants
            for variant in GNN_variant_names:
                score, pred_label = trainer.predict(model_name=method_name, model_path=save_path_best, GNN_variant=variant)
                kendalltau, upset = trainer.evaluate(model_name=method_name, score=score, pred_label=pred_label, GNN_variant=variant)
                kendalltau_full.append(kendalltau)
                upset_full.append(upset)
            
            ### evaluate latest model
            print('\n--- latest model ---')
            kendalltau_full_latest = []
            upset_full_latest = []
            # evaluate for all GNN variants
            for variant in GNN_variant_names:
                score, pred_label = trainer.predict(model_name=method_name, model_path=save_path_latest, GNN_variant=variant)
                kendalltau, upset = trainer.evaluate(model_name=method_name, score=score, pred_label=pred_label, GNN_variant=variant)
                kendalltau_full_latest.append(kendalltau)
                upset_full_latest.append(upset)

        else:
            # evaluate non-GNN methods
            score, pred_label = trainer.predict(model_name=method_name)
            kendalltau_full, upset_full = trainer.evaluate(model_name=method_name, score=score, pred_label=pred_label)
            # non-GNN methods only have one 'best' model, not multiple
            kendalltau_full_latest, upset_full_latest = (kendalltau_full, upset_full)
            

        # append to overall results
        if method_name not in ['DIGRAC', 'ib']:
            kendalltau_res[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full
            kendalltau_res_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full_latest
            final_upset[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += 1
        else:
            kendalltau_res[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full
            kendalltau_res_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full_latest
            final_upset[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += NUM_GNN_VARIANTS
    current_seed_ind += args.num_trials

# print results and save results to arrays
t = Texttable(max_width=120)
t.add_rows([["Parameter", "K", "trainable_alpha", "Fiedler_layer_num", "num_trials", "alpha", "dropout", \
"upset_ratio_coeff", "upset_margin_coeff", "margin","train_with", "baseline", "pretrain with", "pretrain epochs"],
["Values",args.K, args.trainable_alpha, args.Fiedler_layer_num, args.num_trials, args.alpha, args.dropout,
args.upset_ratio_coeff, args.upset_margin_coeff, args.upset_margin, args.train_with, 
args.baseline, args.pretrain_with, args.pretrain_epochs]])
print(t.draw())

for save_dir_name in ['kendalltau', 'upset']:
    if os.path.isdir(os.path.join(dir_name,save_dir_name,method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name,method_str))
        except FileExistsError:
            print('Folder exists for best {}!'.format(save_dir_name))
    if os.path.isdir(os.path.join(dir_name,save_dir_name+'_latest',method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name+'_latest',method_str))
        except FileExistsError:
            print('Folder exists for latest {}!'.format(save_dir_name))

np.save(os.path.join(dir_name,'kendalltau',method_str,save_name), kendalltau_res)
np.save(os.path.join(dir_name,'kendalltau_latest',method_str,save_name), kendalltau_res_latest)
np.save(os.path.join(dir_name,'upset',method_str,save_name), final_upset)
np.save(os.path.join(dir_name,'upset_latest',method_str,save_name), final_upset_latest)


new_shape_upset = (args.num_trials*len(args.seeds), len(compare_names_all), NUM_UPSET_CHOICES)
metric_names = ['test kendall tau', 'test kendall p', 'val kendall tau', 'val kendall p', \
'all kendall tau', 'all kendall p'] + upset_choices
new_shape_kendalltau = (args.num_trials*len(args.seeds), len(compare_names_all), 6)

print_performance_mean_std(args.dataset+'_latest', np.concatenate((kendalltau_res_latest.swapaxes(0,1).reshape(new_shape_kendalltau),
                final_upset_latest.swapaxes(0,1).reshape(new_shape_upset)), axis=-1), 
                compare_names_all, metric_names, False)

print('\n')

print_performance_mean_std(args.dataset+'_best', np.concatenate((kendalltau_res.swapaxes(0,1).reshape(new_shape_kendalltau),
                final_upset.swapaxes(0,1).reshape(new_shape_upset)), axis=-1), 
                compare_names_all, metric_names, False)