cd ../src 

../../parallel -j10 --resume-failed --results ../Output/basketball3_non_proximal --joblog ../joblog/basketball3_non_proximal CUDA_VISIBLE_DEVICES=3 python ./train.py --trainable_alpha --num_trials 10  --hidden 8 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --imbalance_coeff 0 --dataset {5}  --all_methods all_GNNs -All ::: anchor_dist anchor_innerproduct ::: 1 0 ::: 1 0 ::: {2005..2014}  ::: basketball finer_basketball

../../parallel -j10 --resume-failed --results ../Output/basketball3_proximal --joblog ../joblog/basketball3_proximal CUDA_VISIBLE_DEVICES=3 python ./train.py --trainable_alpha --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --imbalance_coeff 0 --dataset {5} --train_with {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2005..2014}  ::: basketball finer_basketball  ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/basketball3_proximal_baseline --joblog ../joblog/basketball3_proximal_baseline CUDA_VISIBLE_DEVICES=3 python ./train.py --trainable_alpha --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --imbalance_coeff 0 --train_with emb_baseline --dataset {5} --cluster_rank_baseline {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2005..2014}  ::: basketball finer_basketball  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS