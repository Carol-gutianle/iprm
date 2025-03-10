from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="response",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False,add_generation_prompt=False)
            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]

    label = data["label"]

    return prompt, chosen, label


class CEDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        print(processed_dataset[0])
        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.resps = processed_dataset["resp"]
        self.labels = processed_dataset["label"]
        self.extra = processed_dataset['extra']

    def process_data(self, data):
        prompt, resp, label = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "resp": resp,
            "label": label,
            'extra':prompt_ids_len,
        }

    def __len__(self):
        length = len(self.resps)
        return length

    def __getitem__(self, idx):
        prompt, chosen, label,extra = self.prompts[idx], self.resps[idx], self.labels[idx], self.extra[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            label,
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        labels = []
        extras = []
        for chosen_id, chosen_mask, label, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            labels.append(label)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        labels = torch.tensor(labels)
        return chosen_ids, chosen_masks, labels, extras



class CEDataset_ICB(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
        batch_size =4,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.batch_size = batch_size

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        processed_dataset = self.batching_data_by_length(processed_dataset)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.resp = processed_dataset["resp"]
        self.label = processed_dataset["label"]
        self.extra = processed_dataset['extra']

    def batching_data(self,dataset):
        i=0
        pre_prompt = None
        final_dataset = {'prompt':[],'resp':[],'label':[],'extra':[]}
        while i < len(dataset):
            if dataset[i]['prompt'] != pre_prompt or len(prompts)>=self.batch_size:
                pre_prompt = dataset[i]['prompt']
                if i!=0:
                    final_dataset['prompt'].append(prompts)
                    final_dataset['resp'].append(resp)
                    final_dataset['label'].append(label)
                    final_dataset['extra'].append(extra)
                prompts, resp, label, extra = [], [], [], []
            prompts.append(dataset[i]['prompt'])
            resp.append(dataset[i]['resp'])
            label.append(dataset[i]['label'])
            extra.append(dataset[i]['extra'])
            i+=1
        return final_dataset

    def batching_data_by_length(self,dataset):
        i=0
        pre_prompt = None
        final_dataset = {'prompt':[],'resp':[],'label':[],'extra':[]}
        while i < len(dataset):
            if dataset[i]['prompt'] != pre_prompt:
                pre_prompt = dataset[i]['prompt']
                if i!=0:
                    final_dataset['prompt'].append(prompts)
                    final_dataset['resp'].append(resp)
                    final_dataset['label'].append(label)
                    final_dataset['extra'].append(extra)
                prompts, resp, label, extra = [], [], [], []
                token_length = 0

            chosen_token = self.tokenizer(
                dataset[i]['prompt']+dataset[i]['resp'],
                return_tensors="pt",
                add_special_tokens=False,
            )

            if token_length + chosen_token["attention_mask"].int().sum().item() > self.max_length:
                final_dataset['prompt'].append(prompts)
                final_dataset['resp'].append(resp)
                final_dataset['label'].append(label)
                final_dataset['extra'].append(extra)
                prompts, resp, label, extra = [dataset[i]['prompt']], [dataset[i]['resp']], [dataset[i]['label']], [dataset[i]['extra']]
                token_length = chosen_token["attention_mask"].int().sum().item()
            else:
                prompts.append(dataset[i]['prompt'])
                resp.append(dataset[i]['resp'])
                label.append(dataset[i]['label'])
                extra.append(dataset[i]['extra'])
                token_length += chosen_token["attention_mask"].int().sum().item()
            i+=1
        return final_dataset

    def process_data(self, data):
        prompt, resp, label = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "resp": resp,
            "label": label,
            'extra':prompt_ids_len,
        }


    def __len__(self):
        length = len(self.resp)
        return length

    def __getitem__(self, idx):
        prompt_list, resp_list, label_list, extra_list = self.prompts[idx], self.resp[idx], self.label[idx], self.extra[idx]
        item_list = []
        for prompt,chosen,label,extra in zip(prompt_list, resp_list, label_list, extra_list):
            chosen = (prompt + chosen).rstrip("\n")
            if not chosen.endswith(self.tokenizer.eos_token):
                chosen += " " + self.tokenizer.eos_token
            chosen_token = self.tokenizer(
                chosen,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            # to avoid EOS_token truncation
            chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            chosen_token["attention_mask"][0][-1] = True

            item_list.append(
                (chosen_token["input_ids"],
                chosen_token["attention_mask"],
                label,
                extra)
            )
        if not self.strategy.args.packing_samples:
            chosen_ids, chosen_masks, reject_ids, rejects_masks, extras = self.batch_collate_fn(item_list)
            return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
        else:
            packed_input_ids, packed_attention_masks, packed_seq_lens,labels, extras = self.batch_packing_collate_fn(item_list)
            return packed_input_ids, packed_attention_masks, packed_seq_lens, labels,extras


    def batch_collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def batch_packing_collate_fn(self, item_list):
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        labels = []
        index = 1
        for chosen_id, chosen_mask, label, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            labels.append(label)
            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens
        labels = torch.tensor(labels)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, labels, extras

    def collate_fn(self, item_list):
        assert len(item_list)==1,item_list
        chosen_ids, chosen_masks, reject_ids, rejects_masks, extras = item_list[0]
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        assert len(item_list)==1,item_list
        packed_input_ids, packed_attention_masks, packed_seq_lens, labels, extras = item_list[0]
        return packed_input_ids, packed_attention_masks, packed_seq_lens, labels,extras





