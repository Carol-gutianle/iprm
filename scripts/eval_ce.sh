#!/bin/bash
set -x

# export PYTHONPATH=/fs-computility/ai-shen/gutianle/iprm

beta=0.05
train_dir=
exp_name=
save_steps=1000
max_epochs=3
LR=5e-7


datapath=
modelpath=
refpretrain=

save_steps=$save_steps
max_epochs=$max_epochs
exp_name=$exp_name

read -r -d '' training_commands <<EOF
openrlhf.cli.eval_ce \
   --save_path /fs-computility/ai-shen/shared/VauAI/gutianle/prime/models/$exp_name \
   --ckpt_path /fs-computility/ai-shen/shared/VauAI/gutianle/prime/models/$exp_name \
   --ref_pretrain $refpretrain \
   --max_ckpt_num 2 \
   --save_steps $save_steps \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --pretrain $modelpath \
   --bf16 \
   --max_epochs $max_epochs \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate $LR \
   --beta $beta \
   --dataset $datapath \
   --apply_chat_template \
   --chosen_key response \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True 
EOF


deepspeed  --module $training_commands