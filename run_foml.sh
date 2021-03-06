DIR=$HOME'/projects/meta-start/'
N_SHOTS=3
N_TEST_SHOTS=3
N_TOTAL=6
N_LAYERS=4
N_REPEATS=10

NAME='foml'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME'inner_1_1_lr003_a' \
       --num_repeats=$N_REPEATS \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=1 \
       --meta_step_size=0.5 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.003 \
       --meta_batch_size=2 \
       --meta_iters=1000 \
       --eval_inner_iters=1 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=$N_LAYERS \
       --hidden_size=32 \
       --n_features=6 \
       --foml \
       --foml_tail=3
