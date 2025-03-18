import os
import json
import random
import argparse
import jsonlines
from datetime import datetime
from tqdm import tqdm
from datasets import interleave_datasets, load_dataset
from openrlhf.datasets import CEMLLMDataset
from openrlhf.models import Actor
from openrlhf.trainer import CETrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from transformers import AutoProcessor
from transformers.trainer import get_scheduler

def load_dataset_from_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def convert_dataset(save_dir, data_path):
    save_path = os.path.join(save_dir, os.path.basename(data_path).replace('.json', '.jsonl'))
    if os.path.exists(save_path):
        return save_path
    data = load_dataset_from_json(data_path)
    writer = jsonlines.Writer(open(save_path, 'w'))
    output_list = []
    for index, sample in tqdm(enumerate(data), desc="Converting"):
        qid = index
        image = sample['image']
        question = sample['question']
        try:
            from PIL import Image
            _ = Image.open(image).convert('RGB')
        except:
            continue
        for sid, step in enumerate(sample['judgment_results']):
            prompt_turn = {
                'role': 'user',
                'content': question
            }
            resp_turn = {
                'role': 'assistant',
                'content': step['step']
            }
            label = step['result']
            output_list.append({
                'response': [prompt_turn, resp_turn],
                'id': f'{qid}_{sid}',
                'image': [image],
                'label': label
            })
    random.shuffle(output_list)
    for d in output_list:
        writer.write(d)
    writer.close()
    return save_path
    
def load_eval_dataset(
    data_path,
    stopping_strategy
):
    eval_data_list = load_dataset('json', data_files=data_path)
    eval_dataset = interleave_datasets(
        [eval_data_list['train']],
        stopping_strategy = stopping_strategy
    )
    return eval_dataset

def evaluate(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)
    # configure processor
    processor = AutoProcessor.from_pretrained(args.ref_pretrain)

    # load weights for ref model
    ref_model = Actor(
        args.ref_pretrain, 
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        packing_samples=args.packing_samples,
    )
    if args.ref_offload:
        ref_model._offload = True
    get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        
    # create converted dataset
    # save_path = convert_dataset("/fs-computility/ai-shen/gutianle/iprm/data", args.dataset)
    save_path = '/fs-computility/ai-shen/shared/VauAI/gutianle/prime/data/eval_0228.json'
    eval_data = load_eval_dataset(save_path, "all_exhausted")
    eval_dataset = CEMLLMDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        processor,
        input_template = args.input_template,
        is_dpo = True,
        multiple_of = args.ring_attn_size        
    )
    print(f"eval_dataset: {len(eval_data)}")
    
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    num_update_steps_per_epoch = len(eval_dataset) // args.train_batch_size
    import math
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        # num_warmup_steps=0,
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    
    
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn
    )
    
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)
    
    trainer = CETrainer(
        model = model,
        ref_model = ref_model,
        tokenizer = tokenizer,
        strategy = strategy,
        optim = optim,
        scheduler = scheduler,
        train_dataloader = eval_dataloader,
        eval_dataloader = eval_dataloader
    )
    
    trainer.evaluate(eval_dataloader, 1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument('--do_test', type=str, default='False', help='Whether to test the model')
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    evaluate(args)