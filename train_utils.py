import os

import torch
from torch.utils.data import Dataset
import utils
import time

from datasets import Dataset as HFDataset

import globals
import pandas as pd
import numpy as np

import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, Mxfp4Config

# from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import tinker
from tinker import types

from peft import get_peft_model, LoraConfig, TaskType
import math

class ScoredDataset(Dataset):
    def __init__(self, df, model_name, tokenizer, max_length=2048):
        self.model_name = model_name
        self.samples = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            row = self.samples.iloc[idx]
            single_row_df = pd.DataFrame([row])
            
            # build prompt str -- used to compute loss mask
            chat_format_msg = utils.build_fewshot_messages(
                single_row_df,
                use_reasoning=True,
                model_name=self.model_name,
            )[0]
            prompt_str = self.tokenizer.apply_chat_template(
                chat_format_msg, tokenize=False, add_generation_prompt=True
            )

            examples = {}
            for key in ['positive', 'negative']:
                reasoning = row[f"{key}_reasoning"].strip()
                answer = row[f"{key}_answer"].strip()

                # begin building the chat format msg, which includes the system prompt and user question
                chat_format_msg = utils.build_fewshot_messages(
                    single_row_df,
                    use_reasoning=True,
                    model_name=self.model_name,
                )[0]
                
                # extend the chat_format_msg with model response. branch based on whether we're using a model in utils.get_reasoning_models() or not
                if self.model_name in utils.get_reasoning_models():
                    assistant_thinking = reasoning
                    assistant_content = f"<answer>{answer}</answer>"
                    chat_format_msg.append({"role": "assistant", "thinking": assistant_thinking, "content": assistant_content})
                else:
                    opener_tag, closer_tag = utils.get_think_tags(self.model_name)
                    assistant_content = f"{opener_tag}{reasoning}{closer_tag}\n\n<answer>{answer}</answer>"
                    chat_format_msg.append({"role": "assistant", "content": assistant_content})
                
                # apply chat template to get a full input/output str
                input_output_str = self.tokenizer.apply_chat_template(
                    chat_format_msg, tokenize=False, add_generation_prompt=False
                )
                tokenized = self.tokenizer(
                    input_output_str,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = tokenized.input_ids.squeeze(0)
                attention_mask = tokenized.attention_mask.squeeze(0)
                
                # Get number of prompt tokens (only tokenize prompt_str)
                prompt_tokenized = self.tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                prompt_len = prompt_tokenized.input_ids.size(1)
                # Construct loss mask: 0s for prompt, 1s for output
                loss_mask = torch.zeros_like(input_ids)
                loss_mask[prompt_len:] = 1

                examples[f"{key}_example"] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,
                    "score": float(row[f"{key}_example_score"]),
                    "missing_data": row[f"{key}_answer"] == "",
                }
                # print what text we're computing the loss over
                # print("FULL INPUT+OUTPUT:", self.tokenizer.decode(input_ids, skip_special_tokens=False))
                # print("LOSS MASK:", loss_mask.tolist())
                # print("PROMPT LEN:", prompt_len, "INPUT LEN:", input_ids.size(0))
                # print("DECODED ALL TOKENS:", self.tokenizer.decode(input_ids, skip_special_tokens=False))
                # print("DECODED PROMPT TOKENS:", self.tokenizer.decode(prompt_tokenized.input_ids.tolist()[0], skip_special_tokens=False))
                # print("DECODED LOSS TOKENS:", self.tokenizer.decode(input_ids[loss_mask==1], skip_special_tokens=False))
                # print("-" * 40)
                # breakpoint()
        except:
            breakpoint()

        return examples
    
    
    def print_examples(self, num_examples=3):
        print(f"Printing {num_examples} examples from the dataset...")
        print("-" * 40)
        for i in range(min(num_examples, len(self.samples))):
            row = self.samples.iloc[i]
            print(f"Example {i}:")
            print("Positive Example:", row['positive_example'])
            print("Positive Score:", row['positive_example_score'])
            print("Negative Example:", row['negative_example'])
            print("Negative Score:", row['negative_example_score'])
            print("-" * 40)

def collate_fn(tokenizer, batch, pad_token_id, max_length=None, fixed_length=None):
    def left_pad_and_stack(tensors, pad_value, max_length=None, fixed_length=None):
        if fixed_length is not None:
            target_len = fixed_length
        elif max_length is not None:
            target_len = min(max(t.size(0) for t in tensors), max_length)
        else:
            target_len = max(t.size(0) for t in tensors)

        padded = []
        for t in tensors:
            pad_len = target_len - t.size(0)
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_value, dtype=t.dtype)
                t = torch.cat([padding, t], dim=0)
            elif pad_len < 0:
                t = t[-target_len:]
            padded.append(t)
        return torch.stack(padded, dim=0)

    # Positive
    pos_input_ids = [ex['positive_example']['input_ids'] for ex in batch]
    pos_attention_mask = [ex['positive_example']['attention_mask'] for ex in batch]
    pos_loss_mask = [ex['positive_example']['loss_mask'] for ex in batch]
    pos_scores = torch.tensor([ex['positive_example']['score'] for ex in batch], dtype=torch.float)

    # Negative
    neg_input_ids = [ex['negative_example']['input_ids'] for ex in batch]
    neg_attention_mask = [ex['negative_example']['attention_mask'] for ex in batch]
    neg_loss_mask = [ex['negative_example']['loss_mask'] for ex in batch]
    neg_scores = torch.tensor([ex['negative_example']['score'] for ex in batch], dtype=torch.float)

    return {
        "pos_input_ids": left_pad_and_stack(pos_input_ids, pad_token_id, max_length=max_length, fixed_length=fixed_length),
        "pos_attention_mask": left_pad_and_stack(pos_attention_mask, 0, max_length=max_length, fixed_length=fixed_length),
        "pos_loss_mask": left_pad_and_stack(pos_loss_mask, 0, max_length=max_length, fixed_length=fixed_length),
        "pos_score": pos_scores,
        "pos_missing_data": torch.tensor([ex['positive_example']['missing_data'] for ex in batch], dtype=torch.long),

        "neg_input_ids": left_pad_and_stack(neg_input_ids, pad_token_id, max_length=max_length, fixed_length=fixed_length),
        "neg_attention_mask": left_pad_and_stack(neg_attention_mask, 0, max_length=max_length, fixed_length=fixed_length),
        "neg_loss_mask": left_pad_and_stack(neg_loss_mask, 0, max_length=max_length, fixed_length=fixed_length),
        "neg_score": neg_scores,
        "neg_missing_data": torch.tensor([ex['negative_example']['missing_data'] for ex in batch], dtype=torch.long),
    }


class CustomTinkerDataset:
    '''
    This is a list with batch sizes that iterates in a random order, a minimal version of Dataset with no collate_fn
    '''
    def __init__(self, data, batch_size, shuffle=True):
        """
        Args:
            data: List of data items or any iterable
            batch_size: Size of each batch
            shuffle: Whether to shuffle the order of batches
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))
        self.rng = np.random.default_rng(0)
        
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        
        for start_idx in range(0, len(self.data), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.data))
            batch_indices = self.indices[start_idx:end_idx]
            batch = [self.data[i] for i in batch_indices]
            yield batch


def move_kwargs_to_gpu(kwargs, device):
    for k,v in kwargs.items():
        if type(v) is torch.Tensor:
            kwargs[k] = v.to(device, non_blocking=True)


def make_dataloader(dataset, model_name, tokenizer, batch_size, max_length=2048, verbose=False):
    dataset = utils.format_dataset(dataset, input_type="positive_example", model_name=model_name)
    dataset = ScoredDataset(
        df=dataset,
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if verbose:
        print(f"Dataset size: {len(dataset)} examples")
        dataset.print_examples(num_examples=3)
        dataset.print_truncated_examples()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(tokenizer, batch, pad_token_id=tokenizer.pad_token_id, max_length=max_length),
        num_workers=0,
        pin_memory=True,
    )


def make_tinker_dataset(dataset, model_name, tokenizer, batch_size, max_length=2048, verbose=False):
    '''
    Returns a CustomTinkerDataset iterable with tinker.types.Datum objects for positive and negative examples, yielding batch sizes of batch_size.
    '''
    hf_dataloader = make_dataloader(
        dataset,
        model_name,
        tokenizer,
        batch_size=1,
        max_length=max_length,
        verbose=verbose,
    )
    list_of_tokenized_data = [
        datapoint for datapoint in hf_dataloader
    ]
    new_datapoints = []
    for tokenized_datapoint in list_of_tokenized_data:
        orig_batch = {k:v for k,v in tokenized_datapoint.items()}
        for key in ["pos", "neg"]:
            input_tokens = tokenized_datapoint[f"{key}_input_ids"]
            tinker_input_tokens = input_tokens[:, :-1].squeeze().tolist()
            tinker_target_tokens = input_tokens[:, 1:].squeeze().tolist()
            weights = tokenized_datapoint[f"{key}_loss_mask"][:, 1:].float().squeeze().tolist()
            tinker_datum = tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(tokens=tinker_input_tokens),
                loss_fn_inputs=dict(weights=weights, target_tokens=tinker_target_tokens)
            )
            orig_batch[f"tinker_{key}_datum"] = tinker_datum
        new_datapoints.append(orig_batch)
    return CustomTinkerDataset(new_datapoints, batch_size=batch_size, shuffle=True)


def compute_masked_loss_per_datapoint(task_model, input_ids, attention_mask, loss_mask, **kwargs):
    '''
    Computes the masked loss for a given task model and input.
    Returns loss per datapoint (and the avg loss per token, per datapoint)
    '''
    outputs = task_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_loss_mask = loss_mask[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    masked_loss = loss_per_token * shift_loss_mask
    nll_loss = masked_loss.sum(axis=1)
    nll_per_token = nll_loss / shift_loss_mask.sum(axis=1)
    return nll_loss, nll_per_token


def lora_wrap_model(model):
    # local model lora hparams
    model_name = utils.model_to_string(model).lower()
    allowed_models = [
        "phi",
        "mistral",
        "llama",
        "qwen",
        "gpt-oss-20b",
    ]
    assert any([x in model_name for x in allowed_models]), f"\nNeed to add LoRA params to peft_config manually -- add exact q_proj and v_proj layer paths to peft_config.target_modules = [paths] from the model: \n{model} \n(SEE MESSAGE ABOVE)"
    if 'phi-4-mini' in model_name:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules=["qkv_proj", "o_proj"],
            bias="none",
        )
    elif "phi-4" in model_name:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            bias="none",
        )
    elif "llama" in model_name:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
            bias="none",
        )
    elif "qwen" in model_name:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
            bias="none",
        )
    elif "gpt-oss-20b" in model_name:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            target_parameters=[
                "7.mlp.experts.gate_up_proj",
                "7.mlp.experts.down_proj",
                "15.mlp.experts.gate_up_proj",
                "15.mlp.experts.down_proj",
                "23.mlp.experts.gate_up_proj",
                "23.mlp.experts.down_proj",
            ],
            # target_parameters=["q_proj", "k_proj", "v_proj", "o_proj",
            #           "gate_proj", "up_proj", "down_proj",]
        )
    print("[LoRA] Wrapping model with with target modules: ", peft_config.target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def summarize_trainable_parameters(model, max_items_print: int = 50):
    """Prints a short summary of total vs trainable parameters and returns a list of trainable param names with sizes.

    This is useful to verify that only LoRA (or other intended) parameters are trainable.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total) * 100 if total > 0 else 0.0
    print(f"Model params: total={total:,}, trainable={trainable:,} ({pct:.4f}%)")

    trainable_named = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_named:
        print("Warning: no trainable parameters found. Did you forget to enable LoRA or set requires_grad?")
        return trainable_named

    print(f"Trainable parameter count: {len(trainable_named)}")
    # Print a compact list (but avoid flooding output)
    to_print = trainable_named[:max_items_print]
    for name, size in to_print:
        print(f"  {name}: {size:,}")
    if len(trainable_named) > max_items_print:
        remaining = len(trainable_named) - max_items_print
        print(f"  ... and {remaining} more trainable parameters (truncated output)")
    return trainable_named


def get_optimizer_and_scheduler(model, learning_rate, quantization, num_training_steps = 1, num_warmup_steps = 0, scheduler = "constant"):
    """Create an optimizer (and LR scheduler) that only includes parameters with requires_grad=True.

    This is the recommended workflow when using LoRA: freeze the base model, let LoRA adapters be trainable,
    and build the optimizer from the trainable parameters only to avoid allocating optimizer state for frozen params.
    """
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("Warning: no trainable parameters found when building optimizer. Falling back to all parameters.")
        trainable_params = list(model.parameters())

    # Choose optimizer based on quantization
    if quantization in ['4bit', '8bit']:
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )

    if scheduler == "constant":
        scheduler_obj = LambdaLR(optimizer, lambda step: 1.0)  # Default to constant LR
    elif scheduler == "cosine":
        scheduler_obj = get_cosine_with_min_lr_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            min_lr_ratio=0.1
        )
    else:
        scheduler_obj = LambdaLR(optimizer, lambda step: 1.0)

    return optimizer, scheduler_obj


def linear_decay_LR(LR, current_step, total_steps):
    return LR * (1 - current_step / total_steps)


def get_cosine_with_min_lr_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Cosine decay with linear warmup, minimum lr at (min_lr_ratio * base_lr).
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Cosine decay to min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, -1)



def save_model_and_tokenizer(model, tokenizer, output_dir, save_name):
    save_path = os.path.join(output_dir, save_name)
    print(f"Saving model to {save_path}...")
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def load_model_and_tokenizer(args, model_name_or_path, quantization, cache_dir, gpu):
    """Load model and tokenizer.

    Args:
        model_name_or_path: model identifier or path
        quantization: quantization string (e.g. '8bit', '16bit', 'none')
        cache_dir: cache dir for transformers
        device: either an int (GPU index or -1 for multi-gpu auto) or a device string like 'cuda:0' or 'cpu'
    """
    if gpu == -1:
        device_str = "auto"
    elif gpu >= 0:
        device_str = f"cuda:{gpu}"

    print(f"Loading model {model_name_or_path} with {quantization} quantization... (device_map={device_str})")
    if quantization == '16bit':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if "phi" in model_name_or_path.lower():
        torch._dynamo.config.capture_scalar_outputs = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        cache_dir=cache_dir
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_load_config = get_quantization_config(model_name_or_path, quantization, cache_dir, device=device_str)
    task_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_load_config,
    ).eval()
    return task_model, tokenizer


def convert_model_quantization_by_reloading(args, task_model, tokenizer, to_quantization, device: str):
    print(f"Converting model precision to {to_quantization} by reloading from disk... (device={device})")
    save_model_and_tokenizer(task_model, tokenizer, args.model_cache_dir, "task_model")
    task_model, tokenizer = load_model_and_tokenizer(
        args,
        os.path.join(args.model_cache_dir, "task_model"),
        to_quantization,
        args.model_cache_dir,
        device,
    )
    return task_model, tokenizer


def get_quantization_config(model_name, quantization, cache_dir, device):
    # device may be a string (e.g. 'cuda:0' or 'cpu') or the special string 'auto'
    # If device == 'auto', return device_map='auto' so transformers will place layers across available devices.
    if device == "auto":
        device_map = "auto"
    else:
        # device is expected to be a string: e.g. 'cuda', 'cuda:0', or 'cpu'
        device_map = {"": device}
    if quantization == '16bit':
        load_config = {
            "torch_dtype": torch.float16,
            "device_map": device_map,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir
        }
    elif quantization in ['4bit', '8bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(quantization == '4bit'),
            load_in_8bit=(quantization == '8bit'),
            bnb_4bit_compute_type=torch.float16,
        )
        if quantization == '4bit':
            bnb_config.quant_type = 'nf4'
        load_config = {
            "quantization_config": bnb_config,
            "device_map": device_map,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir
        }    
    elif quantization == 'mxfp4':
        load_config = {
            "quantization_config": Mxfp4Config(dequantize=False),
            "dtype": torch.bfloat16,
            "device_map": device_map,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir
        }
    elif quantization == 'none':
        load_config = {
            "torch_dtype": "auto",
            "device_map": device_map,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir
        }
    return load_config


async def train_model_local(
    args, task_model, tokenizer, optimizer, scheduler,
    dataset, loss_type, batch_size, grad_accumulation_factor,
    max_length, n_epochs, verbose=False
):
    """
    Local HF model training (matches the local branch of your original function).
    """
    assert 'positive_example' in dataset.columns, "Dataset must contain 'positive_example' column."
    assert loss_type in ['PFT', 'hinge_loss', 'unlikelihood', 'SimPO'], f"Invalid loss type: {loss_type}."

    print(f"  Training model {utils.model_to_string(task_model)} on dataset with {len(dataset)} samples.")
    print(f"  Batch size: {batch_size}, Epochs: {n_epochs}, Grad accumulation: {grad_accumulation_factor}, "
          f"effective batch size: {batch_size * grad_accumulation_factor}, Max length: {max_length}, Loss type: {loss_type}")
    print(f"  Num train steps: {int(np.ceil(len(dataset) / batch_size / grad_accumulation_factor) * n_epochs)}")
    print(f"  LoRA={not args.full_finetuning}, learning rate: {optimizer.defaults['lr']}.")

    time_per_batch_running = []
    time_per_datapoint_running = []
    loss_running = []

    task_model.train()
    device = task_model.device
    torch.cuda.empty_cache()

    if args.print_memory:
        print("\nMEMORY BEFORE TRAINING:")
        utils.print_gpu_memory()

    dataloader = make_dataloader(dataset,
                                 model_name=utils.model_to_string(task_model),
                                 tokenizer=tokenizer,
                                 batch_size=batch_size,
                                 max_length=max_length,
                                 verbose=verbose)

    for epoch in range(n_epochs):
        epoch_loss_running = []
        for i, batch in enumerate(dataloader):
            batch_start = time.time()

            # unpack positive and negative examples
            pos_batch = {
                'input_ids': batch['pos_input_ids'],
                'attention_mask': batch['pos_attention_mask'],
                'loss_mask': batch['pos_loss_mask'],
                'missing_data': batch['pos_missing_data'],
                'pos_score': batch['pos_score'],
            }
            neg_batch = {
                'input_ids': batch['neg_input_ids'],
                'attention_mask': batch['neg_attention_mask'],
                'loss_mask': batch['neg_loss_mask'],
                'missing_data': batch['neg_missing_data'],
                'neg_score': batch['neg_score'],
            }

            # ----- Forward pass -----
            if loss_type == 'PFT':
                non_empty_pos_mask = (pos_batch['missing_data'] == 0).flatten()
                non_empty_pos_batch = {
                    'input_ids': pos_batch['input_ids'][non_empty_pos_mask],
                    'attention_mask': pos_batch['attention_mask'][non_empty_pos_mask],
                    'loss_mask': pos_batch['loss_mask'][non_empty_pos_mask],
                }
                move_kwargs_to_gpu(non_empty_pos_batch, device)

                if pos_batch['missing_data'].sum() == pos_batch['missing_data'].shape[0]:
                    avg_nll_per_token = torch.zeros(pos_batch['input_ids'].shape[0], device=device)
                else:
                    nll_loss, avg_nll_per_token_nonempty = compute_masked_loss_per_datapoint(task_model, **non_empty_pos_batch)
                    avg_nll_per_token = torch.zeros(pos_batch['input_ids'].shape[0], device=device, dtype=avg_nll_per_token_nonempty.dtype)
                    avg_nll_per_token[non_empty_pos_mask] = avg_nll_per_token_nonempty

                losses = avg_nll_per_token
                loss = losses.mean() / grad_accumulation_factor

            if loss_type in ['hinge_loss', 'SimPO', 'unlikelihood']:
                non_empty_pos_mask = (pos_batch['missing_data'] == 0).flatten()
                non_empty_neg_mask = (neg_batch['missing_data'] == 0).flatten()
                non_empty_pos_batch = {
                    'input_ids': pos_batch['input_ids'][non_empty_pos_mask],
                    'attention_mask': pos_batch['attention_mask'][non_empty_pos_mask],
                    'loss_mask': pos_batch['loss_mask'][non_empty_pos_mask],
                }
                non_empty_neg_batch = {
                    'input_ids': neg_batch['input_ids'][non_empty_neg_mask],
                    'attention_mask': neg_batch['attention_mask'][non_empty_neg_mask],
                    'loss_mask': neg_batch['loss_mask'][non_empty_neg_mask],
                }
                move_kwargs_to_gpu(non_empty_pos_batch, device)
                move_kwargs_to_gpu(non_empty_neg_batch, device)

                # FORWARD PASS
                if pos_batch['missing_data'].sum() == pos_batch['missing_data'].shape[0]:
                    pos_avg_nll_per_token = torch.zeros(non_empty_pos_mask.sum(), device=device)
                else:
                    nll_loss, pos_avg_nll_per_token = compute_masked_loss_per_datapoint(task_model, **non_empty_pos_batch)

                if neg_batch['missing_data'].sum() == neg_batch['missing_data'].shape[0]:
                    neg_avg_nll_per_token = torch.zeros(non_empty_neg_mask.sum(), device=device)
                else:
                    nll_loss, neg_avg_nll_per_token = compute_masked_loss_per_datapoint(task_model, **non_empty_neg_batch)

                # scatter back
                filled_out_pos_avg = torch.zeros(len(pos_batch['input_ids']), device=device, dtype=pos_avg_nll_per_token.dtype)
                filled_out_neg_avg = 99*torch.ones(len(neg_batch['input_ids']), device=device, dtype=pos_avg_nll_per_token.dtype)
                if pos_avg_nll_per_token.numel() > 0:
                    filled_out_pos_avg[non_empty_pos_mask] = pos_avg_nll_per_token
                if neg_avg_nll_per_token.numel() > 0:
                    filled_out_neg_avg[non_empty_neg_mask] = neg_avg_nll_per_token

                if loss_type == 'hinge_loss':
                    margin = args.loss_margin
                    zeros = torch.zeros_like(filled_out_pos_avg)
                    losses = torch.maximum(filled_out_pos_avg - filled_out_neg_avg + margin, zeros)
                    loss = losses.mean() / grad_accumulation_factor

                if loss_type == 'SimPO':
                    pos_logp = -filled_out_pos_avg
                    neg_logp = -filled_out_neg_avg
                    beta = args.simpo_beta
                    gamma = args.loss_margin
                    diff = beta * pos_logp - beta * neg_logp - gamma
                    losses = -torch.log(torch.sigmoid(diff))
                    if torch.isinf(losses).any():
                        print("WARNING: Inf values in SimPO loss, skipping these datapoints in the batch.")
                        losses = losses[~torch.isinf(losses)]
                    loss = 1/2 * (filled_out_pos_avg.mean() + losses.mean()) / grad_accumulation_factor

                if loss_type == 'unlikelihood':
                    neg_logp = -filled_out_neg_avg
                    neg_prob = torch.exp(neg_logp)
                    ul_losses = -torch.log(1.0 - neg_prob + 1e-6)
                    pos_rewards = pos_batch['pos_score'].cuda()
                    neg_rewards = neg_batch['neg_score'].cuda()
                    losses = args.unlike_lambda * ((1-neg_rewards) * ul_losses) + pos_rewards * filled_out_pos_avg
                    loss = losses.mean() / grad_accumulation_factor

                avg_nll_per_token = filled_out_pos_avg  # for printing

            # ----- Backward / step -----
            loss.backward()
            step = (i + 1) % grad_accumulation_factor == 0 or (i + 1) == len(dataloader)
            if step:
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            batch_end = time.time()
            time_per_batch_running.append(batch_end - batch_start)
            time_per_datapoint_running.append((batch_end - batch_start) / batch['pos_input_ids'].size(0))
            if verbose and step:
                print(f"Epoch {epoch + 1}/{n_epochs} | Batch {i+1}/{len(dataloader)} | "
                      f"Loss per token (pos data): {avg_nll_per_token.mean().item():.3f} | "
                      f"Time per datapoint: {np.mean(time_per_datapoint_running):.2f}s | "
                      f"Loss: {losses.mean().item():.2f} | "
                      f"Time per batch: {np.mean(time_per_batch_running):.2f}s | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}", end='\n')

            loss_running.append(loss.item())
            epoch_loss_running.append(loss.item())

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {np.mean(epoch_loss_running):.3f} | "
              f"Time per datapoint: {np.mean(time_per_datapoint_running):.2f}s | "
              f"Time per batch: {np.mean(time_per_batch_running):.2f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", end='\n')

        if args.print_memory:
            print("\nMEMORY BETWEEN EPOCHS:")
            utils.print_gpu_memory()

    train_stats = {
        'loss': np.mean(loss_running),
        'steps': int(np.ceil(len(dataloader) / grad_accumulation_factor) * n_epochs),
    }
    task_model.eval()
    return task_model, train_stats


def test_loss_tinker(data: list[types.Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    flat_logprobs = torch.cat(logprobs)
    loss = torch.var(flat_logprobs)
    return loss, {"loss": loss.item()}


async def train_model_tinker(
    args, 
    training_client, 
    task_model,
    tokenizer,
    dataset, loss_type, batch_size, grad_accumulation_factor,
    max_length, n_epochs, verbose=False
):
    """
    Tinker training (matches the Tinker branch of your original function).
    """
    assert 'positive_example' in dataset.columns, "Dataset must contain 'positive_example' column."
    assert loss_type in ['PFT', 'hinge_loss', 'unlikelihood', 'SimPO'], f"Invalid loss type: {loss_type}."

    dataloader = make_tinker_dataset(dataset,
                                    model_name=utils.model_to_string(task_model),
                                    tokenizer=tokenizer,
                                    batch_size=batch_size,
                                    max_length=max_length,
                                    verbose=verbose)

    # filter list of datapoints to those with non-missing positive data if using PFT loss
    if loss_type == 'PFT':
        filtered_data = []
        for datapoint in dataloader.data:
            if datapoint['pos_missing_data'] == 0:
                filtered_data.append(datapoint)
        dataloader = CustomTinkerDataset(filtered_data, batch_size=batch_size, shuffle=True)
        print(f"  Filtered dataset size for PFT loss: {len(filtered_data)} examples.")
    # filter list of datapoints to those with not totally missing pos/neg data if using a contrastive loss
    else:
        filtered_data = []
        for datapoint in dataloader.data:
            if datapoint['pos_missing_data'] == 0 or datapoint['neg_missing_data'] == 0:
                filtered_data.append(datapoint)
        dataloader = CustomTinkerDataset(filtered_data, batch_size=batch_size, shuffle=True)
        print(f"  Filtered dataset size for {loss_type} loss: {len(filtered_data)} examples.")

    print(f"  Training model {utils.model_to_string(task_model)} on dataset with {len(dataloader.data)} samples.")
    print(f"  Batch size: {batch_size}, Epochs: {n_epochs}, Grad accumulation: {grad_accumulation_factor}, "
          f"effective batch size: {batch_size * grad_accumulation_factor}, Max length: {max_length}, Loss type: {loss_type}")
    print(f"  Num train steps: {int(np.ceil(len(dataloader.data) / batch_size / grad_accumulation_factor) * n_epochs)}")
    print(f"  Using Tinker for training model training with lr {args.lr:.1e}")

    time_per_batch_running = []
    time_per_datapoint_running = []
    loss_running = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.print_memory:
        print("\nMEMORY BEFORE TRAINING:")
        utils.print_gpu_memory()

    for epoch in range(n_epochs):
        epoch_loss_running = []
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            cur_bsz = len(batch)

            # TinkerCustomDataset doesn't have a collate_fn, so batch is a list of dicts
            pos_batch = {
                'tinker_data': [batch[j]['tinker_pos_datum'] for j in range(cur_bsz)],
                'missing_data': np.array([batch[j]['pos_missing_data'] for j in range(cur_bsz)]),
                'pos_score': np.array([batch[j]['pos_score'] for j in range(cur_bsz)]),
            }
            neg_batch = {
                'tinker_data': [batch[j]['tinker_neg_datum'] for j in range(cur_bsz)],
                'missing_data': np.array([batch[j]['neg_missing_data'] for j in range(cur_bsz)]),
                'neg_score': np.array([batch[j]['neg_score'] for j in range(cur_bsz)]),
            }

            # ----- Forward pass (PFT only) -----
            if loss_type == 'PFT':
                non_empty_pos_mask = (pos_batch['missing_data'] == 0).flatten()
                if non_empty_pos_mask.sum() == 0:
                    # expand zeros to current batch size
                    avg_nll_per_token = torch.zeros(cur_bsz, device=device)
                else:
                    non_empty_pos_batch = {
                        'tinker_data': [pos_batch['tinker_data'][j] for j in range(cur_bsz) if non_empty_pos_mask[j]],
                    }
                    # queue the forward/backward AND the step
                    fwdbwd_future = await training_client.forward_backward_async(non_empty_pos_batch['tinker_data'], "cross_entropy")
                    optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=args.lr))
                    # now retrieve the results (for logging)
                    fwdbwd_result = await fwdbwd_future
                    optim_result = await optim_future

                    # get nll per example
                    logprobs = [output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs]
                    weights = [example.loss_fn_inputs['weights'].tolist() for example in non_empty_pos_batch['tinker_data']]
                    avg_nll_per_token = [-np.dot(_lp, _w) / sum(_w) for _lp, _w in zip(logprobs, weights)]

                losses = avg_nll_per_token

            elif loss_type in ["SimPO", "unlikelihood"]:
                # join non-empty positive and negative examples
                non_empty_pos_mask = (pos_batch['missing_data'] == 0).flatten()
                non_empty_neg_mask = (neg_batch['missing_data'] == 0).flatten()
                combined_tinker_data = [pos_batch['tinker_data'][j] for j in range(cur_bsz) if non_empty_pos_mask[j]] + \
                                       [neg_batch['tinker_data'][j] for j in range(cur_bsz) if non_empty_neg_mask[j]]
                
                # make metadata for computing contrastive loss
                n_items = cur_bsz
                idx_to_metadata = {}
                idx = 0
                for j in range(cur_bsz):
                    if non_empty_pos_mask[j]:
                        idx_to_metadata[idx] = {'id': j, 'pos_or_neg': 'pos'}
                        idx += 1
                for j in range(cur_bsz):
                    if non_empty_neg_mask[j]:
                        idx_to_metadata[idx] = {'id': j, 'pos_or_neg': 'neg'}
                        idx += 1

                # DEFINE LOSS FUNCTION HERE IN ORDER TO TAKE BATCH METADATA AS VARIABLES IN LOSS FUNCTION WHILE MATCHING TINKER SIGNATURE EXPECTATIONS
                if loss_type == "SimPO":
                    beta, gamma = args.simpo_beta, args.loss_margin
                    def custom_tinker_loss(data: list[types.Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
                        losses = [[0, 0] for _ in range(n_items)]
                        for j in range(len(data)):
                            metadata = idx_to_metadata[j]
                            item_id = metadata['id']
                            pos_or_neg = metadata['pos_or_neg']
                            col_idx = 0 if pos_or_neg == 'pos' else 1
                            datum_logprob = logprobs[j]
                            losses[item_id][col_idx] = -datum_logprob.mean()
                        simpo_loss = 0.0
                        for loss_pair in losses:
                            pos_avg_nll_per_token = loss_pair[0]
                            neg_avg_nll_per_token = loss_pair[1]
                            diff = beta * pos_avg_nll_per_token - beta * neg_avg_nll_per_token - gamma
                            simpo_loss += -torch.log(torch.sigmoid(diff))
                        pos_data_nll = 0.0
                        for loss_pair in losses:
                            pos_data_nll += loss_pair[0]
                        loss = 1/2 * (torch.mean(pos_data_nll) + torch.mean(simpo_loss))
                        return loss, {"loss": loss.item()}

                elif loss_type == "unlikelihood":
                    unlike_lambda = args.unlike_lambda
                    pos_scores = pos_batch['pos_score']
                    neg_scores = neg_batch['neg_score']
                    def custom_tinker_loss(data: list[types.Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
                        # populate pairwise losses
                        pair_nlls = [[0., 0.] for _ in range(n_items)]
                        for j in range(len(data)):
                            metadata = idx_to_metadata[j]
                            item_id = metadata['id']
                            pos_or_neg = metadata['pos_or_neg']
                            col_idx = 0 if pos_or_neg == 'pos' else 1
                            datum_logprob = logprobs[j]
                            pair_nlls[item_id][col_idx] = -datum_logprob.mean()
                        # compute total loss
                        total_loss = 0.0
                        for item_id, loss_pair in enumerate(pair_nlls):
                            pos_avg_nll_per_token = torch.tensor(loss_pair[0]) if isinstance(loss_pair[0], float) else loss_pair[0]
                            neg_avg_nll_per_token = torch.tensor(loss_pair[1]) if isinstance(loss_pair[1], float) else loss_pair[1]
                            neg_prob = torch.exp(-neg_avg_nll_per_token) if neg_avg_nll_per_token != 0.0 else torch.tensor(0.0) # if placeholder value seen (nll of 0), then set prob to 0
                            ul_loss = -torch.log(torch.clamp(1.0 - neg_prob, min=1e-6, max=1.0))
                            pos_reward = torch.tensor(pos_scores[item_id])
                            neg_reward = torch.tensor(neg_scores[item_id])
                            item_loss = pos_reward * pos_avg_nll_per_token + unlike_lambda * ((1 - neg_reward) * ul_loss)
                            total_loss += item_loss
                        loss = total_loss / n_items
                        if loss < 0:
                            breakpoint()
                            print(f"DEBUG: pos_reward={pos_reward.item():.3f}, neg_reward={neg_reward.item():.3f}, pos_nll={pos_avg_nll_per_token.item():.3f}, ul_loss={ul_loss.item():.3f}, item_loss={item_loss.item():.3f}")
                        return loss, {"loss": loss.item()}
                
                # queue the forward/backward AND the step
                async_result = await training_client.forward_backward_custom_async(combined_tinker_data, custom_tinker_loss)
                optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=args.lr))
                # retrieve the results (for logging)
                fwdbwd_result = await async_result
                optim_result = await optim_future
                
                # get custom
                losses = [fwdbwd_result.metrics['loss']]

                # get avg_nll_per_token for positive examples (for logging)
                logprobs = [output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs]
                avg_nll_per_token = [-np.mean(_lp) for _lp in logprobs[:non_empty_pos_mask.sum()]]
            else:
                raise NotImplementedError(f"{loss_type} with Tinker path is not implemented in the current code.")

            loss = np.nanmean(losses)

            # ----- Timing / logging -----
            batch_end = time.time()
            time_per_batch_running.append(batch_end - batch_start)
            time_per_datapoint_running.append((batch_end - batch_start) / cur_bsz)

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} | Batch {i+1}/{len(dataloader)} | "
                      f"Loss per token (pos data): {np.nanmean(avg_nll_per_token).item():.3f} | "
                      f"Time per datapoint: {np.mean(time_per_datapoint_running):.2f}s | "
                      f"Loss: {loss.item():.2f} | "
                      f"Time per batch: {np.mean(time_per_batch_running):.2f}s | "
                      f"LR: {args.lr:.2e}", end='\n')

            loss_running.append(loss.item())
            epoch_loss_running.append(loss.item())

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {np.mean(epoch_loss_running):.3f} | "
              f"NLL per token (pos data): {np.nanmean(avg_nll_per_token).item():.3f} | "
              f"Time per datapoint: {np.mean(time_per_datapoint_running):.2f}s | "
              f"Time per batch: {np.mean(time_per_batch_running):.2f}s | "
              f"LR: {args.lr:.2e}", end='\n')

        if args.print_memory:
            print("\nMEMORY BETWEEN EPOCHS:")
            utils.print_gpu_memory()

    num_batches = int(np.ceil(len(dataloader) / batch_size))
    train_stats = {
        'loss': np.mean(epoch_loss_running),
        'steps': int(np.ceil(num_batches / grad_accumulation_factor) * n_epochs),
    }
    return train_stats


def convert_df_to_trl_dataset(df, tokenizer, max_length=2048, model_name=None, mode="preference", verbose=False):
    """
    Convert a Scored DataFrame into a TRL-compatible dataset.

    Args:
        df (pd.DataFrame): Must contain positive/negative reasoning, answers, and scores.
        tokenizer: Hugging Face tokenizer.
        max_length (int): Max token length for safety.
        model_name (str): Model name, used to determine reasoning tag style.
        mode (str): 'preference' (DPO/SimPO/APO) or 'prompt_completion' (SFT/PFT).

    Returns:
        datasets.Dataset: Hugging Face dataset ready for TRL training.
    """
    assert mode in ["preference", "prompt_completion"], \
        f"Invalid mode '{mode}' — must be 'preference' or 'prompt_completion'."

    dataset_rows = []
    scored_ds = ScoredDataset(df, model_name=model_name, tokenizer=tokenizer, max_length=max_length)
    
    for i in range(len(scored_ds)):
        row = scored_ds.samples.iloc[i]
        single_row_df = pd.DataFrame([row])

        # skip missing data
        missing_data = (row[f"positive_example"] == "")
        if missing_data:
            continue

        # get system and user message, that we will add assistant respond to
        system_and_user_messages = utils.build_fewshot_messages(
                single_row_df,
                use_reasoning=True,
                model_name=model_name,
        )[0]
            
        # Helper to build assistant message correctly depending on model type
        def build_assistant_message(reasoning, answer):
            reasoning = reasoning.strip()
            answer = answer.strip()
            if model_name in utils.get_reasoning_models():
                # Thinking model → separate fields
                return {
                    "role": "assistant",
                    "thinking": reasoning,
                    "content": f"<answer>{answer}</answer>",
                }
            else:
                # Regular model → merge reasoning + answer
                opener_tag, closer_tag = utils.get_think_tags(model_name)
                return {
                    "role": "assistant",
                    "content": f"{opener_tag}{reasoning}{closer_tag}\n\n<answer>{answer}</answer>",
                }
            
        pos_completion = build_assistant_message(row["positive_reasoning"], row["positive_answer"])
        neg_completion = build_assistant_message(row["negative_reasoning"], row["negative_answer"])

        if mode == "prompt_completion":
            dataset_rows.append(
                {
                    "prompt": system_and_user_messages,
                    "completion": [pos_completion],
                    # "score": float(row["positive_example_score"]),
                    "label_type": "positive",
                    "messages": system_and_user_messages + [pos_completion],
                },
            )
        elif mode == "preference":
            assert not row['negative_example'] == "", "Negative example missing but required for preference mode."
            dataset_rows.append({
                "prompt": system_and_user_messages,
                "chosen": [pos_completion],
                "rejected": [neg_completion],
                # "chosen_score": float(row["positive_example_score"]),
                # "rejected_score": float(row["negative_example_score"]),
            })

    hf_dataset = Dataset.from_pandas(pd.DataFrame(dataset_rows))
    print(f"[convert_df_to_trl_dataset] Built {len(hf_dataset)} examples ({mode} mode, "
            f"thinking_model={model_name in utils.get_reasoning_models()}).")
    return hf_dataset


def get_trl_trainer(args, task_model, tokenizer, train_dataset, mode="PFT", verbose=False):
    """
    Build and return a TRL trainer + config pair.

    Args:
        task_model: The model to be fine-tuned.
        tokenizer: Hugging Face tokenizer.
        train_dataset: Hugging Face dataset (from convert_df_to_trl_dataset).
        args: Namespace or dict with hyperparams (learning_rate, train_batch_size, etc.).
        mode: "dpo" (preference-based) or "sft" (prompt-completion-based).

    Returns:
        trainer, trainer_config
    """

    if mode == "APO":
        trainer_config = DPOConfig(
            num_train_epochs=args.epochs,
            loss_type="apo_zero",
            per_device_train_batch_size=args.train_batch_size,
            learning_rate=args.lr,
            beta=0.1,
            max_length=args.train_input_max_size,
            gradient_accumulation_steps=args.grad_accumulation_factor,
            lr_scheduler_type = "constant",
            logging_steps=10,
            output_dir="trl_outputs",
            assistant_only_loss=False,
            report_to = "none", # Use this for WandB etc
        )
        trainer = DPOTrainer(
            model=task_model,
            ref_model=None,  # optional reference model
            tokenizer=tokenizer,
            args=trainer_config,
            train_dataset=train_dataset,
            eval_dataset=None,
        )
    elif mode == "PFT":
        trainer_config = SFTConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            learning_rate=args.lr,
            max_length=args.train_input_max_size,
            gradient_accumulation_steps=args.grad_accumulation_factor,
            lr_scheduler_type = "constant",
            logging_steps=10,
            output_dir="trl_outputs",
            assistant_only_loss=False,
            report_to = "none", # Use this for WandB etc
        )
        trainer = SFTTrainer(
            model=task_model,
            args=trainer_config,
            train_dataset=train_dataset,
            eval_dataset=None,
        )
    else:
        raise ValueError(f"Invalid mode '{mode}' for trl")

    print(f"\n[build_trl_trainer] Built {mode.upper()} trainer with LR={trainer_config.learning_rate}, "
          f"data_size={len(train_dataset)}, epochs={trainer_config.num_train_epochs}, "
          f"batch_size={trainer_config.per_device_train_batch_size}, "
          f"beta={getattr(trainer_config, 'beta', None)}")

    return trainer

