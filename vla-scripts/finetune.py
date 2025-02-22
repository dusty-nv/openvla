"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""
import os
import sys
import time
import json
import pprint
import signal
import datetime
import threading

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from copy import deepcopy

import draccus
import torch
import torch.distributed as dist
import numpy as np
import bitsandbytes as bnb
import tqdm
import contextlib

from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics, AttributeDict
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS, rlds_dataset_builder_transform

from prismatic.vla.sim.mimicgen import MGStreamingDataset
from prismatic.util.grokfast import gradfilter_ma, gradfilter_ema

from merge import merge_lora


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

          
@dataclass
class FinetuneConfig:
    # fmt: off
    exp_id: str = None                                              # Unique experiment ID (will be initialized if left None)
    exp_tag: str = None                                             # Extra tag to end onto the end of experiment ID string
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = None                                    # Temporary directory for LoRA weights before fusing

    # Sim settings
    tasks: List[str] = field(default_factory=list)                  # List of MimicGen tasks to simulate
    task_weights: List[str] = field(default_factory=list)           # Task selection frequency distribution
    camera: str = "agentview"                                       # Camera to use (agentview, frontview, robot0-eye_in_hand)
    save_dir: Path = None                                           # Directory to dump sample generated episodes to (optional)
    save_freq: int = None                                           # Save every N generated episodes for viewing (optional)
    
    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    epochs: int = None                                              # Number of training passes through dataset (overrides max_steps)
    max_images: int = None                                          # Max number of training frames/images (overrides max_steps)
    max_steps: int = 200_000                                        # Max number of fine-tuning steps (gradient accumulation)
    save_steps: int = 5000                                          # Interval for checkpoint saving
    metric_steps: int = 8                                           # The number of batches to average loss/accuracy metrics over
    resume_step: int = 0                                            # Global Step to Resume (should match checkpoint)
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    grad_filter: str = None                                         # Use 'ema' or 'ma' to enable grokfast 

    #optim_bits: int = 32                                            # 32-bit or 8-bit precision for AdamW solver
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    tensorboard_logdir: str = "/data/logs/tensorboard"
    
    # fmt: on
       
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    #torch.distributed.init_process_group(backend='gloo')
    distributed_state = PartialState(backend='gloo')
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    if not cfg.exp_id:
        data_exp = cfg.dataset_name
        if cfg.tasks:
            data_exp = data_exp + '+' + '+'.join(cfg.tasks)
        cfg.exp_id = (
            f"{cfg.vla_path.split('/')[-1].split('+')[0]}+{data_exp}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            cfg.exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.use_quantization:
            cfg.exp_id += "+q-4bit"
        if cfg.exp_tag:
            cfg.exp_id += f"+{cfg.exp_tag}"
        cfg.exp_id += f"+{datetime.datetime.now().strftime('%y%m%d_%H%M')}"
     
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`  ({cfg.exp_id})")
   
    # Start =>> Build Directories
    if not cfg.adapter_tmp_dir:
        cfg.adapter_tmp_dir = cfg.run_root_dir / 'adapters'
        
    cfg.run_root_dir = cfg.run_root_dir / cfg.exp_id
    cfg.adapter_tmp_dir = cfg.adapter_tmp_dir / cfg.exp_id

    os.makedirs(cfg.run_root_dir, exist_ok=True)
    os.makedirs(cfg.adapter_tmp_dir, exist_ok=True)
    
    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate) #bnb.optim.AdamW(trainable_params, lr=cfg.learning_rate, optim_bits=cfg.optim_bits)
    grads = None
        
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset => RLDS/OXE or simulated
    if cfg.dataset_name.lower() == 'mimicgen':
        vla_dataset = MGStreamingDataset(
            cfg.tasks,
            cfg.task_weights,
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle=bool(cfg.shuffle_buffer_size > 0),
            image_aug=cfg.image_aug,
            camera=cfg.camera,
            save_dir=cfg.save_dir,
            save_freq=cfg.save_freq,
        )
    else:
        if cfg.dataset_name not in OXE_DATASET_CONFIGS:
            data_cfg = deepcopy(OXE_DATASET_CONFIGS['rlds_dataset_builder'])
            OXE_DATASET_CONFIGS[cfg.dataset_name] = data_cfg
            
        if cfg.dataset_name not in OXE_STANDARDIZATION_TRANSFORMS:
            OXE_STANDARDIZATION_TRANSFORMS[cfg.dataset_name] = rlds_dataset_builder_transform
            
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        )
        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )

    # Simulated datasets don't have a length because they're streamed forever
    try:
        simulated = bool(len(vla_dataset) <= 0)
    except:
        simulated = True

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, cfg.run_root_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    if cfg.epochs:
        if simulated:
            raise ValueError(f"cannot train for number of epochs when simulator is used")
        cfg.max_steps = int(len(vla_dataset) * cfg.epochs / cfg.batch_size / cfg.grad_accumulation_steps) #int(len(dataloader) * cfg.epochs / cfg.grad_accumulation_steps)
 
    if cfg.max_images:
        cfg.max_steps = int(cfg.max_images / cfg.batch_size / cfg.grad_accumulation_steps)
         
    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        cfg.tensorboard_logdir = os.path.join(cfg.tensorboard_logdir, cfg.exp_id)
        tensorboard = SummaryWriter(log_dir=cfg.tensorboard_logdir)

    if simulated:
        print(f"\nSimulator tasks {cfg.tasks}  (batch_size={cfg.batch_size}, grad_accumulation_steps={cfg.grad_accumulation_steps})\n\n{pprint.pformat(cfg, indent=2)}\n")
    else:
        print(f"\nDataset frames {len(vla_dataset):,} => batches {len(dataloader):,} => steps {len(dataloader)//cfg.grad_accumulation_steps:,} (batch_size={cfg.batch_size}, grad_accumulation_steps={cfg.grad_accumulation_steps})\n\n{pprint.pformat(cfg, indent=2)}\n")
    
    # Store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    metrics = AttributeDict(
        loss = Metric('loss', 'Loss/logits'),
        loss_action = Metric('loss_action', 'Loss/action'),
        token_accuracy = Metric('token_accuracy', 'Accuracy/tokens'),
        action_accuracy = Metric('action_accuracy', 'Accuracy/action'),
    )
    
    for metric in metrics.values():
        metric.resize(cfg.metric_steps, cfg.grad_accumulation_steps)

    # Keep filling data until the requested number of steps is reached
    def next_batch():
        batch_idx = 0
        while True:
            for batch in dataloader:
                if batch_idx / cfg.grad_accumulation_steps >= cfg.max_steps:
                    return
                yield batch_idx, batch
                batch_idx += 1
          
    # Allow the user to interrupt training with Ctrl+C
    interrupts = ProcessInterrupt()
    time_begin = time.perf_counter()
    
    # Train!
    with tqdm.tqdm(initial=cfg.resume_step, total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in next_batch():
            if interrupts:
                break
                
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
                
            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            metrics.token_accuracy += correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
            continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))

            metrics.loss_action += torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            metrics.action_accuracy += 1.0 - metrics.loss_action[-1] 

            if progress.n > 10 and loss >= metrics.loss.mean() * 1.3:
                print(f"Step {progress.n}, abnormal loss detected:  loss={loss}  avg={metrics.loss.mean()}")
                dump_step(
                    progress.n, cfg,
                    metrics=dict(step=progress.n, loss=float(loss), prev_loss=metrics.loss.mean(), token_accuracy=metrics.token_accuracy.history[-1], action_accuracy=metrics.action_accuracy.history[-1]),
                    inputs=batch,
                    outputs=dict(action_preds=action_preds, action_gt=action_gt, continuous_actions_pred=continuous_actions_pred, continuous_actions_gt=continuous_actions_gt)
                )
                    
            metrics.loss += loss
            
            # Optimizer Step
            if batch_idx == 0 or batch_idx % cfg.grad_accumulation_steps != 0:
                continue
   
            if cfg.grad_filter == 'ema':
                grads = gradfilter_ema(vla, grads=grads, alpha=0.98, lamb=2.0)
            elif cfg.grad_filter == 'ma':
                grads = gradfilter_ma(vla, grads=grads, window_size=100, lamb=lamb)
            elif cfg.grad_filter:
                raise ValueError(f"grad_filter should be 'ema' or 'ma' (was {cfg.grad_filter})")
             
            if cfg.grad_filter and progress.n == 0:
                print(f"using grad_filter {cfg.grad_filter}")
                  
            optimizer.step()
            optimizer.zero_grad()
                
            progress.set_description(f"{metrics.loss}  {metrics.token_accuracy}  {metrics.action_accuracy}")
            progress.update() # increments progress.n (global step count)
            print('')
            
            if distributed_state.is_main_process:
                for metric in metrics.values():
                    tensorboard.add_scalar(metric.tensorboard, metric.step_mean(), progress.n)

            if progress.n % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    save_checkpoint(vla, processor, cfg, progress.n, metrics)
                dist.barrier()
                    
        # print training stats
        train_time = time.perf_counter() - time_begin
        train_frames = progress.n * cfg.batch_size * cfg.grad_accumulation_steps
        train_rate = train_frames / train_time
        
        print(f"\nDone training after {progress.n} steps, {train_frames} frames  ({int(train_time)} seconds, {train_rate:.2f} fps)")
        
        # save final checkpoint and merge LoRA  
        if distributed_state.is_main_process:
            save_checkpoint(vla, processor, cfg, progress.n, metrics)
            del vla  # reduce memory to be able to load LoRA
            torch.cuda.empty_cache()
            merge_lora(cfg.vla_path, cfg.adapter_tmp_dir, cfg.run_root_dir)  
            
        dist.barrier()
        

def save_checkpoint(vla, processor, cfg, steps, metrics):
    # by default, only last checkpoint is kept, continually overwriting it!
    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
    save_dir = cfg.adapter_tmp_dir if cfg.use_lora else cfg.run_root_dir
    print(f"Saving Model Checkpoint for Step {steps} under {save_dir}")
    processor.save_pretrained(cfg.run_root_dir)
    vla.module.save_pretrained(save_dir)
    save_metrics(metrics, steps, cfg)

     
def save_metrics(metrics, steps, cfg):
    for save_dir in [cfg.run_root_dir, cfg.adapter_tmp_dir]:
        stats_path = save_dir / "train_statistics.json"
        stats_past = []
        
        try:
            with open(stats_path, 'r') as f:
                stats_past = json.load(f)
        except Exception:
            pass
        
        stats = {
            'steps': steps, 
            'frames': steps * cfg.batch_size * cfg.grad_accumulation_steps,   
        }
        
        for metric in metrics.values():
            stats[metric.name] = {
                'step': metric.step_mean(),     
                'mean': metric.mean(),
            }

        try:
            with open(stats_path, 'w') as f:
                json.dump(stats_past + [stats], f, indent=2)
        except Exception as error:
            print(f"Exception while saving training stats to {stats_path}\n  {error}")
            
    print(f"Saving training stats to {stats_path}")
        

def dump_step(step, cfg, metrics, inputs, outputs):
    for save_dir in [cfg.run_root_dir, cfg.adapter_tmp_dir]:
        save_dir = save_dir / "debug"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"rank={torch.distributed.get_rank()}_step={step}.json"
        try:
            for k,v in inputs.items(): 
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    inputs[k] = v = v.tolist()
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], bytes):
                    for vi, vb in enumerate(v):
                        v[vi] = vb.decode("utf-8")
            for k,v in outputs.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    outputs[k] = v = v.tolist()
            with open(save_path, 'w') as f:
                json.dump(dict(
                    metrics=metrics,
                    inputs=inputs,
                    outputs=outputs,
                ), f, indent=2)
        except Exception as error:
            print(f"Exception while saving training stats to {save_path}\n  {error}")
    print(f"Saved debug dump to {save_path}")
                   
class Metric:
    # Accumulate / average training stats
    def __init__(self, name, tensorboard=None, window=10, step_window=None):
        self.name = name
        self.tensorboard = tensorboard
        self.resize(window, step_window)
     
    def __str__(self):
        return f"{self.name}={self.step_mean():.4f} ~{self.mean():.4f}"
          
    def __len__(self):
        return len(self.history)
    
    def __getitem__(self, index):
        return self.history[index]
     
    def __iadd__(self, value):
        self.append(value)
        return self
        
    def append(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.history.append(value)
           
    def mean(self, window=None):
        if window and window <= len(self.history):
            history = [self.history[-x-1] for x in range(window)] #self.history[-window:]
        else:
            history = self.history
        return sum(history) / len(history)      
  
    def step_mean(self):
        return self.mean(self.step_window)
        
    def resize(self, window=10, step_window=None):
        self.history = deque(maxlen=window)
        self.window = window
        self.step_window = step_window
   
   
class ProcessInterrupt(threading.Thread):
    # Ctrl+D interrupt handler
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interrupts = 0
        self.start()
    
    def __len__(self):
        return self.interrupts
            
    def run(self):
        print(">> Press Ctrl+D to save weights and stop training early\n")
        while True:
            try:
                input()
            except EOFError:
                self.interrupts += 1
                if self.interrupts > 2:
                    print("\nTerminating training process early\n")
                    sys.exit(0)
                elif self.interrupts > 1:
                    print("\nPress Ctrl+D again for ungraceful termination\n")
                else:
                    print("\nCtrl+D pressed, interrupting training...\n")
 
          
if __name__ == "__main__":
    finetune()
