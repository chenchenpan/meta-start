DIR='/Users/crazydonkey/projects/meta-start/'

N_SHOTS=5
N_TOTAL=6

NAME='hypertile'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=1 \
       --meta_step_size=0.1 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.001 \
       --meta_batch_size=2 \
       --meta_iters=2000 \
       --eval_inner_iters=1 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=1 \
       --hidden_size=32 \
       --use_cat_embed \
       --cat_embed_path=$DIR'data/preprocess_data/category_embeddings.json' \
       --use_hypernet \
       --n_features=6
