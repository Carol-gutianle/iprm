#!/bin/bash

#SBATCH --job-name=yty_debug
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --nodelist=g53
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --ntasks=1
#SBATCH --output=%j.out
#SBATCH --account=test


set -x

export PYTHONPATH=$PYTHONPATH:`realpath .`


beta=0.05
train_dir=dpo_data
exp_name=dpo
save_steps=-1
max_epochs=1
LR=5e-7


datapath=./processed_data/$train_dir
modelpath=meta-llama/Llama-3.1-8B-Instruct

save_steps=$save_steps
max_epochs=$max_epochs
timestamp=$(date '+%Y%m%d-%H%M%S');
exp_name=$exp_name


read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoints/$exp_name \
   --ckpt_path ./checkpoints/$exp_name \
   --max_ckpt_num 2 \
   --save_steps $save_steps \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain $modelpath \
   --bf16 \
   --max_epochs $max_epochs \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate $LR \
   --beta $beta \
   --dataset $datapath \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing
EOF


deepspeed --module $training_commands