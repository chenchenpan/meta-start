DIR=$HOME'/projects/meta-start/'
N_REPEATS=10
N_SHOTS=2
N_TOTAL=3

NAME='baseline'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME \
       --num_repeats=$N_REPEATS \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=0 \
       --meta_step_size=0.5 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.001 \
       --meta_batch_size=2 \
       --meta_iters=0 \
       --eval_inner_iters=50 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=3 \
       --hidden_size=64 \
       --n_features=6


NAME='joint'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME \
       --num_repeats=$N_REPEATS \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=1 \
       --meta_step_size=0.5 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.001 \
       --meta_batch_size=2 \
       --meta_iters=1000 \
       --eval_inner_iters=0 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=1 \
       --hidden_size=32 \
       --n_features=6

NAME='foml'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME \
       --num_repeats=$N_REPEATS \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=5 \
       --meta_step_size=0.5 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.001 \
       --meta_batch_size=2 \
       --meta_iters=1000 \
       --eval_inner_iters=5 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=1 \
       --hidden_size=32 \
       --n_features=6 \
       --foml \
       --foml_tail=1

NAME='reptile'
python run_experiment.py \
       --data_path=$DIR'data/preprocess_data/splitted_data_pos_'$N_TOTAL'.json' \
       --output_dir=$DIR'experiments/'$NAME \
       --num_repeats=$N_REPEATS \
       --seed=0 \
       --num_shots=$N_SHOTS \
       --num_train_shots=$N_SHOTS \
       --inner_iters=5 \
       --meta_step_size=0.5 \
       --meta_step_size_final=0.1 \
       --learning_rate=0.001 \
       --meta_batch_size=2 \
       --meta_iters=1000 \
       --eval_inner_iters=5 \
       --num_eval_samples=100 \
       --eval_interval=100 \
       --weight_decay=1 \
       --n_layers=1 \
       --hidden_size=32 \
       --n_features=6

