#* RACE
RACE: albert-base-v2
python -u run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints --task_name race --per_gpu_eval_batch_size 4 \
--logging_steps 1 --max_seq_length 512 --model_type albert --model_name_or_path albert-base-v2 \
--data_dir RACE --learning_rate 1e-5 --num_train_epochs 1 --output_dir albert_base_race \
--per_gpu_train_batch_size 4 --gradient_accumulation_steps 1 --warmup_steps 1000 --save_steps 3000

RACE: bert-base-uncased
python -u run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints --task_name race --per_gpu_eval_batch_size 4 \
--logging_steps 1 --max_seq_length 512 --model_type bert --model_name_or_path bert-base-uncased \
--data_dir RACE --learning_rate 1e-5 --num_train_epochs 1 --output_dir bert_base_race \
--per_gpu_train_batch_size 4 --gradient_accumulation_steps 1 --warmup_steps 1000 --save_steps 3000

RACE: xlnet-base-cased 
python -u run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints --task_name race --per_gpu_eval_batch_size 4 \
--logging_steps 1 --max_seq_length 512 --model_type xlnet --model_name_or_path xlnet-base-cased \
--data_dir RACE --learning_rate 1e-5 --num_train_epochs 1 --output_dir xlnet_base_race \
--per_gpu_train_batch_size 4 --gradient_accumulation_steps 1 --warmup_steps 1000 --save_steps 3000

#* DREAM
DREAM: albert-base-v2 
python run_multiple_choice.py --do_lower_case  --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name dream --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type albert --model_name_or_path albert-base-v2 \
--data_dir DREAM --learning_rate 1e-5 --num_train_epochs 3 --output_dir albert_base_dream \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382

DREAM: bert-base-uncased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name dream --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type bert --model_name_or_path bert-base-uncased \
--data_dir DREAM --learning_rate 1e-5 --num_train_epochs 3 --output_dir bert_base_dream \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382

DREAM: xlnet-base-cased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name dream --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type xlnet --model_name_or_path xlnet-base-cased \
--data_dir DREAM --learning_rate 1e-5 --num_train_epochs 3 --output_dir xlnet_base_dream \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382

#* MCTest
MCTest500: albert-base-v2
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type albert --model_name_or_path albert-base-v2 \
--data_dir MCTest500 --learning_rate 1e-5 --num_train_epochs 2 --output_dir albert_base_mctest \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382


MCTest500: bert-base-uncased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type bert --model_name_or_path bert-base-uncased \
--data_dir MCTest500 --learning_rate 1e-5 --num_train_epochs 2 --output_dir bert_base_mctest \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382


MCTest500: xlnet-base-cased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type xlnet --model_name_or_path xlnet-base-cased \
--data_dir MCTest500 --learning_rate 1e-5 --num_train_epochs 2 --output_dir xlnet_base_mctest \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382

MCTest160: albert-base-v2
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type albert --model_name_or_path albert-base-v2 \
--data_dir MCTest160 --learning_rate 1e-5 --num_train_epochs 2 --output_dir albert_base_mctest160 \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382


MCTest160: bert-base-uncased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type bert --model_name_or_path bert-base-uncased \
--data_dir MCTest160 --learning_rate 1e-5 --num_train_epochs 2 --output_dir bert_base_mctest160 \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382


MCTest160: xlnet-base-cased
python run_multiple_choice.py --do_lower_case --do_train --do_eval --do_test \
--overwrite_output --eval_all_checkpoints  --task_name mctest --per_gpu_eval_batch_size=4 \
--logging_steps 1 --max_seq_length 512 --model_type xlnet --model_name_or_path xlnet-base-cased \
--data_dir MCTest160 --learning_rate 1e-5 --num_train_epochs 2 --output_dir xlnet_base_mctest160 \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps 1 --warmup_steps 100 --save_steps 382


