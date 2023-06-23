import os
import time
from datetime import datetime
from typing import Union, Tuple

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import kendalltau, rankdata
import scipy.sparse as sp
from tqdm.auto import tqdm # automatically use command-line or notebook version

# internal files
from .utils import write_log, scipy_sparse_to_torch_sparse, get_powers_sparse
from .metrics import calculate_upsets
from .GNN_models import DIGRAC_Ranking, DiGCN_Inception_Block_Ranking
from .preprocess import load_data
from .get_adj import get_second_directed_adj
from .SpringRank import SpringRank
from .comparison import syncRank_angle, syncRank, serialRank, btl, davidScore, eigenvectorCentrality, PageRank, rankCentrality, mvr
from .comparison import SVD_RS, SVD_NRS, serialRank_matrix
from .param_parser import ArgsNamespace


class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args: ArgsNamespace, random_seed: int, save_name_base: str, adj = None):
        """
        Constructing the trainer instance.
        
        Parameters
        ----------
        - args: Arguments object
        - ...
        - adj: If provided, use this adjacency matrix instead of loading data from disk
        """
        self.args = args
        self.device = args.device
        self.random_seed = random_seed
        
        self.GNN_variant_names = ['dist', 'innerproduct', 'proximal_dist', 'proximal_innerproduct', 'proximal_baseline']
        self.NUM_GNN_VARIANTS = len(self.GNN_variant_names) # number of GNN variants for each architecture

        self.upset_choices = ['upset_simple', 'upset_ratio', 'upset_naive']
        self.NUM_UPSET_CHOICES = len(self.upset_choices)

        self.GNN_model_names = ['DIGRAC', 'ib']
        self.NON_GNN_model_names = ['SpringRank', 'btl', 'davidScore', 'eigenvectorCentrality',
                                    'PageRank', 'rankCentrality', 'syncRank', 'mvr', 'SVD_RS', 'SVD_NRS']

        self.label, self.train_mask, self.val_mask, self.test_mask, self.features, self.A = load_data(args, random_seed, adj=adj)
        if not self.args.be_silent: print('loaded test_mask:', self.test_mask)
        self.features = torch.FloatTensor(self.features).to(self.device)
        self.args.N = self.A.shape[0]
        self.A_torch = torch.FloatTensor(self.A.toarray()).to(self.device)
        
        self.nfeat = self.features.shape[1]
        if self.label is not None:
            self.label = torch.LongTensor(self.label).to(self.device)
            self.label_np = self.label.to('cpu')
            if  self.args.dataset[:3] != 'ERO':
                self.args.K = int(self.label_np.max() - self.label_np.min() + 1)
        else:
            self.label_np = None
        self.num_clusters = self.args.K

        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        save_name = save_name_base + 'Seed' + str(random_seed)

        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, date_time)

        self.splits = self.args.num_trials
        if self.test_mask is not None and self.test_mask.ndim == 1:
            self.train_mask = np.repeat(
                self.train_mask[:, np.newaxis], self.splits, 1)
            self.val_mask = np.repeat(
                self.val_mask[:, np.newaxis], self.splits, 1)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)
        
        if not self.args.load_only:
            if os.path.isdir(self.log_path) == False:
                try:
                    os.makedirs(self.log_path)
                except FileExistsError:
                    print('Folder exists!')

            write_log(vars(args), self.log_path)  # write the settings

    def init_model(self, model_name: str, A: Union[sp.spmatrix, None]=None):
        #-> tuple[DIGRAC_Ranking | DiGCN_Inception_Block_Ranking | Unbound, tuple[Tensor, Tensor] | Unbound | None, tuple[Tensor, Tensor] | Unbound | None, Any | Unbound | None, Any | Unbound | None]

        if A is None: A = self.A

        if model_name == 'DIGRAC':
            norm_A = get_powers_sparse(A, hop=1, tau=self.args.tau)[
                1].to(self.device)
            norm_At = get_powers_sparse(A.transpose(), hop=1, tau=self.args.tau)[
                1].to(self.device)
            edge_index, edge_weights = (None, None)
        elif model_name == 'ib':
            #edge_index = torch.LongTensor(A.nonzero())
            coo = A.tocoo()
            edge_index = torch.LongTensor(np.array((coo.row, coo.col)))
            edge_weights = torch.FloatTensor(A.data)
            edge_index1 = edge_index.clone().to(self.device)
            edge_weights1 = edge_weights.clone().to(self.device)
            edge_index2, edge_weights2 = get_second_directed_adj(edge_index, self.features.shape[0],self.features.dtype,
            edge_weights)
            edge_index2 = edge_index2.to(self.device)
            edge_weights2 = edge_weights2.to(self.device)
            edge_index = (edge_index1, edge_index2)
            edge_weights = (edge_weights1, edge_weights2)
            del edge_index2, edge_weights2
            norm_A, norm_At = (None, None)
        
        #for split in range(self.splits): # TODO: handle splits
        
        ### get baseline scores
        if self.args.baseline == 'mvr': raise NameError('Cannot use mvr for baseline model')
        score, pred_label = self.predict(model_name=self.args.baseline)

        score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.device)
        if score.min() < 0:
            score_torch = torch.sigmoid(score_torch)
        upset1 = calculate_upsets(self.A_torch, score_torch)
        upset2 = calculate_upsets(torch.transpose(self.A_torch, 0, 1), score_torch)
        
        if upset1.detach().item() > upset2.detach().item():
            score = -score
        score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.device)

        ### initialize NN models
        if model_name == 'DIGRAC':
            model = DIGRAC_Ranking(num_features=self.nfeat, dropout=self.args.dropout, hop=self.args.hop, fill_value=self.args.tau, 
        embedding_dim=self.args.hidden*2, Fiedler_layer_num=self.args.Fiedler_layer_num, alpha=self.args.alpha, 
        trainable_alpha=self.args.trainable_alpha, initial_score=score_torch, prob_dim=self.num_clusters, sigma=self.args.sigma).to(self.device)
        elif model_name == 'ib':
            model = DiGCN_Inception_Block_Ranking(num_features=self.nfeat, dropout=self.args.dropout,
        embedding_dim=self.args.hidden*2, Fiedler_layer_num=self.args.Fiedler_layer_num, alpha=self.args.alpha, 
        trainable_alpha=self.args.trainable_alpha, initial_score=score_torch, prob_dim=self.num_clusters, sigma=self.args.sigma).to(self.device)
        
        return model, edge_index, edge_weights, norm_A, norm_At


    def train(self, model_name: str, split: int=0):
        #################################
        # training and evaluation
        #################################
        if model_name not in self.GNN_model_names:
            raise ValueError('Can only train GNN models using DIGRAC or ib')
        
        if self.args.upset_ratio_coeff + self.args.upset_margin_coeff == 0:
            raise ValueError('Incorrect loss combination!')
        
        #args = self.args
        #A = scipy_sparse_to_torch_sparse(self.A).to(self.device)
        
        model, edge_index, edge_weights, norm_A, norm_At = self.init_model(model_name, A=self.A)

        ### initialize optimizer
        if self.args.optimizer == 'Adam':
            opt = optim.Adam(model.parameters(), lr=self.args.lr,
                            weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            opt = optim.SGD(model.parameters(), lr=self.args.lr,
                            weight_decay=self.args.weight_decay)
        else:
            raise NameError('Please input the correct optimizer name, Adam or SGD!')
        M = self.A_torch

        ### initialize the masks
        if self.test_mask is not None:
            train_index = self.train_mask[:, split]
            val_index = self.val_mask[:, split]
            test_index = self.test_mask[:, split]
            if self.args.AllTrain:
                # to use all nodes
                train_index[:] = True
                val_index[:] = True
                test_index[:] = True
            #train_A = scipy_sparse_to_torch_sparse(self.A[train_index][:, train_index]).to(self.device)
            #val_A = scipy_sparse_to_torch_sparse(self.A[val_index][:, val_index]).to(self.device)
            #test_A = scipy_sparse_to_torch_sparse(self.A[test_index][:, test_index]).to(self.device)
        else:
            train_index = np.ones(self.args.N, dtype=bool)
            val_index = train_index
            test_index = train_index
            #train_A = scipy_sparse_to_torch_sparse(self.A).to(self.device)
            #val_A = train_A
            #test_A = train_A
        #################################
        # Train/Validation/Test
        #################################
        best_val_loss = 1000.0
        early_stopping = 0
        log_str_full = ''
        train_with = self.args.train_with
        if self.args.pretrain_with == 'serial_similarity' and self.args.train_with[:8] == 'proximal':
            serialRank_mat = serialRank_matrix(self.A[train_index][:, train_index])
            serialRank_mat = serialRank_mat/max(0.1, serialRank_mat.max())
            serial_matrix_train = torch.FloatTensor(serialRank_mat.toarray()).to(self.device)
        
        last_time = time.time() # slow down updating the postfix of tqdm
        
        with tqdm(range(self.args.epochs), unit='epochs') as tqdm_bar:
            for epoch in tqdm_bar:

                if self.args.optimizer == 'Adam' and epoch == self.args.pretrain_epochs and self.args.train_with[:8] == 'proximal':
                    opt = optim.SGD(model.parameters(), lr=10*self.args.lr,
                                    weight_decay=self.args.weight_decay)
                start_time = time.time()
                ####################
                # Train
                ####################

                model.train()
                if model_name == 'DIGRAC':
                    _ = model(norm_A, norm_At, self.features)
                elif model_name == 'ib':
                    _ = model(edge_index, edge_weights, self.features)
                if train_with == 'dist' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                    score = model.obtain_score_from_dist()
                elif train_with == 'innerproduct' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                    score = model.obtain_score_from_innerproduct()
                else:
                    score = model.obtain_score_from_proximal(train_with[9:])

                if self.args.upset_ratio_coeff > 0:
                    train_loss_upset_ratio = calculate_upsets(M[train_index][:,train_index], score[train_index])               
                else:
                    train_loss_upset_ratio = torch.ones(1, requires_grad=True).to(self.device)
                if self.args.upset_margin_coeff > 0:
                    train_loss_upset_margin = calculate_upsets(M[train_index][:,train_index], score[train_index], style='margin', margin=self.args.upset_margin)               
                else:
                    train_loss_upset_margin = torch.ones(1, requires_grad=True).to(self.device)

                train_loss = self.args.upset_ratio_coeff * train_loss_upset_ratio + self.args.upset_margin_coeff * train_loss_upset_margin
                if self.args.pretrain_with == 'serial_similarity' and epoch < self.args.pretrain_epochs and self.args.train_with[:8] == 'proximal':
                    pretrain_outside_loss = torch.mean((model.obtain_similarity_matrix()[train_index][:, train_index] - serial_matrix_train) ** 2)
                    train_loss += pretrain_outside_loss
                    outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f}, pretrian outside loss: {:6f},'.format(train_loss.detach().item(),
                    train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item(), 
                    pretrain_outside_loss.detach().item())
                else:
                    outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(train_loss.detach().item(),
                    train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item())
                opt.zero_grad()
                try:
                    train_loss.backward()
                except RuntimeError:
                    log_str = '{} trial {} RuntimeError!'.format(model_name, split)
                    log_str_full += log_str + '\n'
                    print(log_str)
                    if not os.path.isfile(self.log_path + '/'+model_name+'_model'+str(split)+'.t7'):
                            torch.save(model.state_dict(), self.log_path +
                            '/'+model_name+'_model'+str(split)+'.t7')
                    torch.save(model.state_dict(), self.log_path +
                            '/'+model_name+'_model_latest'+str(split)+'.t7')
                    break
                opt.step()
                ####################
                # Validation
                ####################
                model.eval()

                if model_name == 'DIGRAC':
                    _ = model(norm_A, norm_At, self.features)
                elif model_name == 'ib':
                    _ = model(edge_index, edge_weights, self.features)
    
                if train_with == 'dist' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                    score = model.obtain_score_from_dist()
                elif train_with == 'innerproduct' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                    score = model.obtain_score_from_innerproduct()
                else:
                    score = model.obtain_score_from_proximal(train_with[9:])

                if self.args.upset_ratio_coeff > 0:
                    val_loss_upset_ratio = calculate_upsets(M[val_index][:,val_index], score[val_index])               
                else:
                    val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(self.device)
                if self.args.upset_margin_coeff > 0:
                    val_loss_upset_margin = calculate_upsets(M[val_index][:,val_index], score[val_index], style='margin', margin=self.args.upset_margin)               
                else:
                    val_loss_upset_margin = torch.ones(1, requires_grad=True).to(self.device)

                val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin


                outstrval = 'val loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(val_loss.detach().item(),
                val_loss_upset_ratio.detach().item(), val_loss_upset_margin.detach().item())

                duration = "---, {:.4f}, seconds ---".format(time.time() - start_time)
                log_str = ("{}, / {} epoch,".format(epoch, self.args.epochs)) + outstrtrain + outstrval + duration
                log_str_full += log_str + '\n'

                if (time.time() - last_time > 0.5): # slow down updating the postfix of tqdm
                    tqdm_bar.set_postfix_str(f"train loss: {train_loss.detach().item():.3f}, val loss: {val_loss.detach().item():.3f}")
                    last_time = time.time()

                ####################
                # Save weights
                ####################
                save_perform = val_loss.detach().item()
                if save_perform <= best_val_loss:
                    early_stopping = 0
                    best_val_loss = save_perform

                    if os.path.isdir(self.log_path) == False:
                        try:
                            os.makedirs(self.log_path)
                        except FileExistsError:
                            print('Folder exists!')
                    save_path_best = self.log_path +'/'+model_name+'_model'+str(split)+'.t7'
                    torch.save(model.state_dict(), save_path_best) # save the best model
                else:
                    early_stopping += 1
                if early_stopping > self.args.early_stopping:
                    tqdm_bar.close()
                    if not self.args.be_silent:
                        print(f'Early stopped after {self.args.early_stopping} epochs without improvement.')
                    break
        
        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')
        save_path_latest = self.log_path + '/' + model_name + '_model_latest' + str(split) + '.t7'
        torch.save(model.state_dict(), save_path_latest) # save the latest model

        status = 'w'
        if os.path.isfile(self.log_path + '/'+model_name+'_log'+str(split)+'.csv'):
            status = 'a'
        with open(self.log_path + '/'+model_name+'_log'+str(split)+'.csv', status) as file:
            file.write(log_str_full)
            file.write('\n')
            status = 'a'

        torch.cuda.empty_cache()
        return save_path_best, save_path_latest # return the paths of the two models


    def evaluation(self, logstr, score, A_torch, label_np, val_index, test_index, SavePred, save_path, split, identifier_str):
        kendalltau_full = np.zeros((3, 2))
        kendalltau_full[:] = np.nan
        if score.min().detach().item() < 0:
            if score.min().detach().item() > -1:
                score = (score + 1)/2
            else:
                score = torch.sigmoid(score)
        upset1 = calculate_upsets(A_torch, score)
        upset2 = calculate_upsets(torch.transpose(A_torch, 0, 1), score)
        if upset1.detach().item() < upset2.detach().item():
            upset_ratio = upset1
        else:
            upset_ratio = upset2
            score = -score
        pred_label = rankdata(-score.detach().cpu().numpy(), 'min')
        upset_simple = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.device), style='simple').detach().item()
        upset_naive = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.device), style='naive').detach().item()
        upset_full = [upset_simple, upset_ratio.detach().item(), upset_naive]
        if SavePred:
            np.save(save_path+identifier_str+'_pred'+str(split), pred_label)
            np.save(save_path+identifier_str+'_scores'+str(split), score.detach().cpu().numpy())

        #logstr += '\n From ' + identifier_str + ': '
        logstr += ', '
        if label_np is not None:
            # test
            tau, p_value = kendalltau(pred_label[test_index], label_np[test_index])
            outstrtest = f'Test kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[0] = [tau, p_value]

            # val
            tau, p_value = kendalltau(pred_label[val_index], label_np[val_index])
            outstrval = f'Validation kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[1] = [tau, p_value]


            # all
            tau, p_value = kendalltau(pred_label, label_np)
            outstrall = f'All kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[2] = [tau, p_value]

            logstr += outstrtest + outstrval + outstrall
        logstr += f'upset simple: {upset_simple:.6f}, upset ratio: {upset_ratio.detach().item():.6f}, upset naive: {upset_naive:.6f}'
        return logstr, upset_full, kendalltau_full


    def predict(self, model_name: str, model_path: Union[str, None]=None,
                A: Union[sp.spmatrix, None]=None, GNN_variant: Union[str, None]=None):

        # allow prediction for data loaded in different Trainer
        if A is None: A = self.A

        if model_name in self.NON_GNN_model_names:
            score, pred_label = self.predict_non_nn(model_name, A)
        
        elif model_name in self.GNN_model_names:
            if not model_path:
                raise ValueError('Need to supply a path to a model for NN prediction (DIGRAC, ib).')
            score, pred_label = self.predict_nn(model_name, model_path, A, GNN_variant=GNN_variant)
        
        else:
            raise NameError(f'Please input the correct model name from:\
                SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                PageRank, rankCentrality, mvr, SVD_RS, SVD_NRS, DIGRAC, ib, instead of {model_name}!')
        
        return score, pred_label
    

    def evaluate(self, model_name, score, pred_label, split=0, GNN_variant=None):

        if model_name in self.NON_GNN_model_names:
            kendalltau_full, upset_full = self.evaluate_non_nn(model_name, score, pred_label, split)
        
        elif model_name in self.GNN_model_names:
            kendalltau_full, upset_full = self.evaluate_nn(model_name, score, pred_label, split, GNN_variant)

        else:
            raise NameError(f'Please input the correct model name from:\
                SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                PageRank, rankCentrality, mvr, SVD_RS, SVD_NRS, DIGRAC, ib, instead of {model_name}!')

        return kendalltau_full, upset_full


    def predict_nn(self, model_name, model_path, A, split=0, GNN_variant=None):

        # unless otherwise specified, predict the way it was trained: dist, innerproduct, or proximal_[dist, innerproduct, baseline]
        if GNN_variant is None: GNN_variant = self.args.train_with

        ### initialize NN model
        # note that for proximal prediction, we need initial scores
        model, edge_index, edge_weights, norm_A, norm_At = self.init_model(model_name, A=A)
        
        ### load the model state
        model.load_state_dict(torch.load(model_path))

        ### set into evaluation mode
        model.eval()

        ### load data into model
        if model_name == 'DIGRAC':
            _ = model(norm_A, norm_At, self.features)
        elif model_name == 'ib':
            _ = model(edge_index, edge_weights, self.features)

        ### obtain prediction
        if GNN_variant == 'dist':
            score_model = model.obtain_score_from_dist()
        elif GNN_variant == 'innerproduct':
            score_model = model.obtain_score_from_innerproduct()
        else: # GNN_variant ~ 'proximal ...'
            score_model = model.obtain_score_from_proximal(GNN_variant[9:]) # proximal 'dist', 'innerproduct', or 'baseline'

        score = score_model
        pred_label = None

        torch.cuda.empty_cache()
        return score, pred_label
    
    def evaluate_nn(self, model_name, score, A=None, split=0, GNN_variant=None):

        if A is None: A = self.A
        A_torch = torch.FloatTensor(A.toarray()).to(self.device)

        ### initialize the masks
        if self.test_mask is not None:
            train_index = self.train_mask[:, split]
            val_index = self.val_mask[:, split]
            test_index = self.test_mask[:, split]
            if self.args.AllTrain:
                # to use all nodes
                train_index[:] = True
                val_index[:] = True
                test_index[:] = True
        else:
            train_index = np.ones(self.args.N, dtype=bool)
            val_index = train_index
            test_index = train_index

        ### get the upsets
        if self.args.upset_ratio_coeff > 0:
            val_loss_upset_ratio = calculate_upsets(A_torch[val_index][:,val_index], score[val_index]) 
            test_loss_upset_ratio = calculate_upsets(A_torch[test_index][:,test_index], score[test_index])  
            all_loss_upset_ratio = calculate_upsets(A_torch, score)                
        else:
            val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(self.device)
            test_loss_upset_ratio = torch.ones(1, requires_grad=True).to(self.device)
            all_loss_upset_ratio = torch.ones(1, requires_grad=True).to(self.device)
        if self.args.upset_margin_coeff > 0:
            val_loss_upset_margin = calculate_upsets(A_torch[val_index][:,val_index], score[val_index], style='margin', margin=self.args.upset_margin) 
            test_loss_upset_margin = calculate_upsets(A_torch[test_index][:,test_index], score[test_index], style='margin', margin=self.args.upset_margin)               
            all_loss_upset_margin = calculate_upsets(A_torch, score, style='margin', margin=self.args.upset_margin)                             
        else:
            val_loss_upset_margin = torch.ones(1, requires_grad=True).to(self.device)
            test_loss_upset_margin = torch.ones(1, requires_grad=True).to(self.device)
            all_loss_upset_margin = torch.ones(1, requires_grad=True).to(self.device)

        ### calculate the losses
        val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin
        test_loss = self.args.upset_ratio_coeff * test_loss_upset_ratio + self.args.upset_margin_coeff * test_loss_upset_margin
        all_loss = self.args.upset_ratio_coeff * all_loss_upset_ratio + self.args.upset_margin_coeff * all_loss_upset_margin

        logstr = f'Results for {model_name} {GNN_variant}:\n'
        logstr += f' val loss: {val_loss.detach().item():.3f}, test loss: {test_loss.detach().item():.3f}, all loss: {all_loss.detach().item():.3f}'
        #print(logstr)

        ### calculate kendalltau and upset
        # (the last two dimensions) rows: test, val, all; cols: kendall tau, kendall p value
        #kendalltau_full = np.zeros([self.NUM_GNN_VARIANTS, self.splits, 3, 2])
        kendalltau_full = np.zeros([self.splits, 3, 2])
        kendalltau_full[:] = np.nan

        #upset_full = np.zeros([self.NUM_GNN_VARIANTS, self.splits, self.NUM_UPSET_CHOICES])
        upset_full = np.zeros([self.splits, self.NUM_UPSET_CHOICES])
        upset_full[:] = np.nan

        base_save_path = self.log_path + '/'+model_name
        #if no other GNN variant is given, evaluate the way it was trained: dist, innerproduct, or proximal_[dist, innerproduct, baseline]
        if GNN_variant is None: GNN_variant = self.args.train_with
        logstr, upset_full[split], kendalltau_full[split] = self.evaluation(logstr, score, A_torch, self.label_np, val_index, test_index,
                                                                            self.args.SavePred, base_save_path, split, GNN_variant)
        
        print(logstr)

        #with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', status) as file:
        #    file.write(logstr)
        #    file.write('\n')

        torch.cuda.empty_cache()
        return kendalltau_full, upset_full


    def predict_non_nn(self, model_name, A):

        score = None
        pred_label = None

        if model_name == 'SpringRank':
            score = SpringRank(A,alpha=0,l0=1,l1=1)
        elif model_name == 'serialRank':
            score = serialRank(A)
        elif model_name == 'btl':
            score = btl(A)
        elif model_name == 'davidScore':
            score = davidScore(A)
        elif model_name == 'eigenvectorCentrality':
            score = eigenvectorCentrality(A)
        elif model_name == 'PageRank':
            score = PageRank(A)
        elif model_name == 'rankCentrality':
            score = rankCentrality(A)
        elif model_name == 'syncRank':
            pred_label = syncRank(A)
            score = syncRank_angle(A) # scores
        elif model_name == 'mvr':
            pred_label = mvr(A)
        elif model_name == 'SVD_RS':
            score = SVD_RS(A)
        elif model_name == 'SVD_NRS':
            score = SVD_NRS(A)
        
        return score, pred_label
    

    def evaluate_non_nn(self, model_name, score, pred_label, A=None, split=0):

        if A is None: A = self.A
        A_torch = torch.FloatTensor(self.A.toarray()).to(self.device)
        
        # rows: test, val, all; cols: kendall tau, kendall p value
        kendalltau_full = np.zeros([self.splits, 3, 2])
        kendalltau_full[:] = np.nan
        upset_full = np.zeros([self.splits, self.NUM_UPSET_CHOICES])
        upset_full[:] = np.nan

        logstr = ''

        if self.test_mask is not None:
            val_index = self.val_mask[:, split]
            test_index = self.test_mask[:, split]
            if self.args.AllTrain:
                # to use all nodes
                val_index[:] = True
                test_index[:] = True
        else:
            val_index = np.ones(self.args.N, dtype=bool)
            test_index = val_index

        if model_name not in ['mvr']:
            score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.device)
            if score.min() < 0:
                if score.min()  > -1:
                    score_torch = (score_torch + 1)/2
                else:
                    score_torch = torch.sigmoid(score_torch)
            upset1 = calculate_upsets(A_torch, score_torch)
            upset2 = calculate_upsets(torch.transpose(A_torch, 0, 1), score_torch)
            
            if model_name not in ['syncRank']:
                if upset1.detach().item() > upset2.detach().item():
                    upset_ratio = upset2.detach().item()
                    score = -score
                else:
                    upset_ratio = upset1.detach().item()
                pred_label = rankdata(-score, 'min')
            else:
                if upset1.detach().item() < upset2.detach().item():
                    upset_ratio = upset1.detach().item()
                    pred_label = 1 + pred_label.max()-pred_label
                else:
                    upset_ratio = upset2.detach().item()
        else:
            upset_ratio = np.nan
            score = np.nan

        upset_simple = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.device), style='simple').detach().item()
        upset_naive = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.device), style='naive').detach().item()
        upset_full[split] = [upset_simple, upset_ratio, upset_naive]

        if self.args.SavePred:
            np.save(self.log_path + '/'+model_name+
                    '_pred'+str(split), pred_label)
            np.save(self.log_path + '/'+model_name+
                    '_scores'+str(split), score)
        
        print('Results for {}:'.format(model_name))

        if self.label is not None:
            # test
            tau, p_value = kendalltau(pred_label[test_index], self.label_np[test_index])
            outstrtest = f'Test kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[split, 0] = [tau, p_value]
            
            # val
            tau, p_value = kendalltau(pred_label[val_index], self.label_np[val_index])
            outstrval = f'Validation kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[split, 1] = [tau, p_value]
            
            
            # all
            tau, p_value = kendalltau(pred_label, self.label_np)
            outstrall = f'All kendall tau: {tau:.3f}, kendall p value: {p_value:.3f}, '
            kendalltau_full[split, 2] = [tau, p_value]
                
            logstr += outstrtest + outstrval + outstrall

        logstr += f'upset simple: {upset_simple:.6f}, upset ratio: {upset_ratio:.6f}, upset naive: {upset_naive:.6f}'
        print(logstr)

        #with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', 'a') as file:
        #    file.write(logstr)
        #    file.write('\n')

        return kendalltau_full, upset_full