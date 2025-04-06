import contextlib

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import time
import argparse
import logging
import os
import math
import sys
import traceback
# +++ Import necessary schedulers +++
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# --- End Import ---
from tqdm import tqdm

# Ensure models imports from the correct location relative to train.py
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# Debug print function
def print_debug(rank, msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[DEBUG {timestamp} Rank-{rank}] {msg}", flush=True)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    # ... (EMA function remains the same) ...
    is_ddp = isinstance(model, DDP)
    model_params = OrderedDict(model.module.named_parameters() if is_ddp else model.named_parameters())
    ema_params = OrderedDict(ema_model.named_parameters())
    model_keys = set(model_params.keys())
    ema_keys = set(ema_params.keys())
    if model_keys != ema_keys:
         print(f"Warning: EMA key mismatch! Model keys: {len(model_keys)}, EMA keys: {len(ema_keys)}")
    for name, param in model_params.items():
        if name in ema_params:
             ema_params[name].mul_(decay).add_(param.data.detach(), alpha=1 - decay)


def requires_grad(model, flag=True):
    # ... (remains the same) ...
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    # ... (remains the same) ...
    if dist.is_available() and dist.is_initialized():
        print("[Cleanup] Debug: Calling dist.destroy_process_group()")
        dist.destroy_process_group()
        print("[Cleanup] Debug: dist.destroy_process_group() finished.")

def create_logger(logging_dir):
    # ... (remains the same) ...
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_rank_zero = (rank == 0)
    logger = logging.getLogger(f"{__name__}_rank{rank}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'[[34m%(asctime)s[0m Rank-{rank}] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if is_rank_zero and logging_dir:
        try:
            os.makedirs(logging_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Rank 0 logger setup with file logging in: {logging_dir}")
        except Exception as e:
            print(f"Error setting up file logger for Rank 0: {e}")
    logger.propagate = False
    print_debug(rank, "Logger instance created.")
    return logger

def center_crop_arr(pil_image, image_size):
    # ... (remains the same) ...
    try:
        pil_image = pil_image.convert('RGB')
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        cropped_arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
        if cropped_arr.shape[0] != image_size or cropped_arr.shape[1] != image_size:
             pil_image = pil_image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)
             return pil_image
        return Image.fromarray(cropped_arr)
    except Exception as e:
        print(f"Error during center_crop_arr: {e}. Returning black image.")
        return Image.new('RGB', (image_size, image_size), color = 'black')

# +++ VALIDATION FUNCTION (remains the same) +++
@torch.no_grad()
def validate_one_epoch(model, vae, diffusion, loader, device, rank, epoch, logger):
    # ... (validation logic is unchanged) ...
    model.eval(); vae.eval()
    total_val_loss = 0.0; val_steps = 0
    is_ddp = dist.is_available() and dist.is_initialized()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Validation", disable=(rank != 0))
    for i, batch_data in enumerate(pbar):
        if not batch_data: continue
        try: x, _ = batch_data; x = x.to(device, non_blocking=True)
        except Exception as e: logger.warning(f"Skipping val batch {i}: {e}"); continue
        try:
            latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (latent_x.shape[0],), device=device)
            model_kwargs = dict()
            loss_dict = diffusion.training_losses(model, latent_x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            if not torch.isfinite(loss): logger.warning(f"Non-finite val loss: {loss.item()}"); continue
            if is_ddp:
                 loss_tensor = loss.detach().clone(); dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG); reduced_loss_item = loss_tensor.item()
            else: reduced_loss_item = loss.item()
            total_val_loss += reduced_loss_item; val_steps += 1
            if rank == 0: avg_loss = total_val_loss / val_steps if val_steps > 0 else 0; pbar.set_postfix({"AvgValLoss": f"{avg_loss:.4f}", "Steps": val_steps})
        except Exception as e: logger.error(f"Val step {i} rank {rank} error: {e}\n{traceback.format_exc()}")
    model.train()
    if val_steps == 0: logger.warning(f"Epoch {epoch} Validation: No steps completed."); return float('inf')
    avg_val_loss = total_val_loss / val_steps; return avg_val_loss

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # ... (DDP Setup - world_size is defined here) ...
    rank_for_debug = int(os.environ.get('RANK', 0))
    print_debug(rank_for_debug, "Entering main function.")
    assert torch.cuda.is_available(), "Training requires GPU."
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    print_debug(rank_for_debug, f"DDP check: is_ddp = {is_ddp}")
    if is_ddp:
        rank = int(os.environ["RANK"]); world_size = int(os.environ['WORLD_SIZE']); local_rank = int(os.environ['LOCAL_RANK']); device = local_rank
        print_debug(rank_for_debug, f"Rank {rank} (local {local_rank}), World {world_size}. Initializing DDP...")
        try: dist.init_process_group(backend='nccl', init_method='env://'); torch.cuda.set_device(device); dist.barrier()
        except Exception as e: print_debug(rank_for_debug, f"!!! DDP Init failed: {e}"); raise
    else:
        rank = 0; world_size = 1; local_rank = 0; device = 0 # <<< world_size defined here for single process
        print_debug(rank_for_debug, f"Single process mode. Device: {device}")
    torch.cuda.set_device(device)


    # --- Batch size configuration ---
    # Moved the divisibility check here, after world_size is known
    if args.global_batch_size % (world_size * args.accumulation_steps) != 0:
         # Use logger if available, otherwise print
         msg = f"Global batch size ({args.global_batch_size}) must be divisible by world_size*accumulation_steps ({world_size}*{args.accumulation_steps}). Exiting."
         try: logger.error(msg);
         except NameError: print(f"!!! ERROR: {msg}")
         if is_ddp: cleanup() # Attempt cleanup if DDP was initialized
         sys.exit(1) # Exit script

    assert args.global_batch_size % world_size == 0, f"Global batch size {args.global_batch_size} must be divisible by world size {world_size}."
    per_proc_batch_size = int(args.global_batch_size // world_size)

    # Calculate micro_batch_size based on per_proc_batch_size
    # This was previously done in the logging print, do it explicitly here
    micro_batch_size = per_proc_batch_size // args.accumulation_steps
    effective_batch_size = args.global_batch_size # Effective size per optim step is the global size
    print_debug(rank_for_debug, f"Global BS: {args.global_batch_size}, Accum steps: {args.accumulation_steps}, WorldSize: {world_size}, PerProc BS: {per_proc_batch_size}, Micro-BS per proc: {micro_batch_size}")
    # --- Seeding ---
    seed = args.global_seed + rank; torch.manual_seed(seed); np.random.seed(seed)
    # --- Experiment Dirs & Logger ---
    experiment_dir = None; checkpoint_dir = None
    if rank == 0:
         # ... (Experiment dir creation logic remains the same) ...
         os.makedirs(args.results_dir, exist_ok=True)
         try: experiment_index = max([int(d.split('-')[0]) for d in os.listdir(args.results_dir) if d.split('-')[0].isdigit()] + [-1]) + 1
         except Exception: experiment_index = 0
         model_string_name = args.model.replace("/", "-"); attn_str = f"attn-{args.attention_type}" + (f"-w{args.window_size}" if args.attention_type == 'swa' else ""); uncond_str = "uncond"
         experiment_dir = os.path.join(args.results_dir, f"{experiment_index:03d}-{model_string_name}-{attn_str}-{uncond_str}")
         checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
         os.makedirs(experiment_dir, exist_ok=True); os.makedirs(checkpoint_dir, exist_ok=True)
    if is_ddp: dist.barrier()
    logger = create_logger(experiment_dir if rank == 0 else None)
    if rank == 0: # Log config
        logger.info("-------------------- Configuration --------------------")
        for arg_name, value in vars(args).items(): logger.info(f"{arg_name}: {value}")
        logger.info(f"World Size: {world_size}"); logger.info(f"Global Batch Size: {args.global_batch_size}"); logger.info(f"Accumulation Steps: {args.accumulation_steps}"); logger.info(f"Micro-batch Size Per Proc: {micro_batch_size}")
        logger.info(f"Experiment Directory: {experiment_dir}"); logger.info("------------------------------------------------------")

    # --- Best Validation Loss Tracking ---
    best_val_loss = float('inf'); best_val_epoch = 0; epochs_no_improve = 0; early_stop_triggered_local = False
    if rank == 0: logger.info(f"Initializing best validation loss tracker. Patience: {args.patience}")

    # --- Model Setup ---
    # ... (Model config, DiT instantiation, EMA creation remain the same) ...
    assert args.image_size % 8 == 0; latent_size = args.image_size // 8; window_size = args.window_size if args.attention_type == 'swa' else None
    model_config = {'input_size': latent_size, 'num_classes': 0, 'window_size': window_size}
    try: model = DiT_models[args.model](**model_config).to(device)
    except Exception as e: logger.error(f"Model Instantiation Error: {e}"); cleanup(); sys.exit(1)
    ema = deepcopy(model).to(device); requires_grad(ema, False)
    if is_ddp: # DDP Wrapping
        try: model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.find_unused_parameters)
        except Exception as e: logger.error(f"DDP Error: {e}"); cleanup(); sys.exit(1)

    # --- Diffusion, VAE, Optimizer ---
    diffusion = create_diffusion(timestep_respacing="")
    vae_model_name = f"stabilityai/sd-vae-ft-{args.vae}"
    try: vae = AutoencoderKL.from_pretrained(vae_model_name).to(device); requires_grad(vae, False); vae.eval()
    except Exception as e: logger.error(f"VAE Load Error: {e}"); cleanup(); sys.exit(1)
    if rank == 0: # Log param count
        num_params = sum(p.numel() for p in model.parameters()); num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total DiT Params: {num_params:,}, Trainable: {num_params_trainable:,}")
    # +++ Optimizer Definition (ADJUST FOR BETAS HERE if desired) +++
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999)) # Default betas
    # Example change: betas=(0.9, 0.98)
    # --- End Optimizer ---
    print_debug(rank_for_debug, "Optimizer created.")

    # --- Data Loading ---
    # +++ Data Augmentation Control (DISABLE/MODIFY HERE) +++
    train_transform_list = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    # Example: Add ColorJitter
    # train_transform_list.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    train_transform = transforms.Compose(train_transform_list)
    # --- End Augmentation ---
    val_transform = transforms.Compose([ # Validation transform usually simpler
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # ... (Dataset loading, path checks remain the same) ...
    train_data_path = args.data_path; train_dir = os.path.join(train_data_path, 'train'); val_dir = os.path.join(train_data_path, 'val')
    if not os.path.isdir(train_dir): logger.error(f"Train dir not found: '{train_dir}'"); cleanup(); sys.exit(1)
    has_val_data = os.path.isdir(val_dir)
    if not has_val_data: logger.warning(f"Val dir not found: '{val_dir}'. Val/EarlyStop disabled."); args.patience = 0
    try:
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform) if has_val_data else None
        train_len = len(train_dataset); val_len = len(val_dataset) if val_dataset else 0
        print_debug(rank_for_debug, f"Datasets loaded. Train: {train_len}, Val: {val_len}.")
        if train_len == 0: logger.error(f"Training dataset empty."); cleanup(); sys.exit(1)
        # Calculate steps based on *optimizer steps* per epoch
        steps_per_epoch_optim = math.ceil(train_len / args.global_batch_size) # Number of micro-batches / accum_steps
        effective_epochs_for_scheduler = args.scheduler_tmax_epochs if args.scheduler_tmax_epochs > 0 else args.epochs
        total_optim_steps_for_scheduler = steps_per_epoch_optim * effective_epochs_for_scheduler
        # Calculate interval checkpoint frequency based on optimizer steps
        # Calculate total *planned* optimizer steps
        total_planned_optim_steps = steps_per_epoch_optim * args.epochs
        ckpt_every = args.ckpt_every if args.ckpt_every > 0 else max(1, total_planned_optim_steps // 10) # ~10 checkpoints if 0
        print_debug(rank_for_debug, f"Micro-batches/epoch: {math.ceil(train_len / per_proc_batch_size)}. Optim Steps/epoch: {steps_per_epoch_optim}. Total Planned Optim Steps: {total_planned_optim_steps}. Scheduler T_max Optim Steps: {total_optim_steps_for_scheduler}. Interval Ckpt Freq: {ckpt_every} optim steps.")
    except Exception as e: logger.error(f"Dataset Load Error: {e}"); cleanup(); sys.exit(1)

    # --- LR Scheduler with Warmup ---
    try:
        # Main decay scheduler (Cosine) - T_max is total decay steps *after* warmup
        decay_steps = total_optim_steps_for_scheduler - args.warmup_steps
        if decay_steps <= 0:
             logger.warning(f"Warmup steps ({args.warmup_steps}) >= total scheduler steps ({total_optim_steps_for_scheduler}). Disabling decay.")
             # Option 1: Just use warmup (constant LR after)
             # main_scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1) # Placeholder
             # Option 2: Use cosine over 1 step (effectively constant) - Safer?
             main_scheduler = CosineAnnealingLR(opt, T_max=1, eta_min=args.lr if args.warmup_steps > 0 else 1e-6) # Keep target LR if warmed up
             decay_steps = 1 # Avoid issues with SequentialLR
        else:
             main_scheduler = CosineAnnealingLR(opt, T_max=decay_steps, eta_min=1e-6)

        if args.warmup_steps > 0:
            # Linear warmup scheduler
            # Calculate start factor (avoiding division by zero if lr=0)
            start_lr = 1e-7 # Start warmup from a very small value
            start_factor = start_lr / args.lr if args.lr > 0 else 0.0
            warmup_scheduler = LinearLR(opt, start_factor=start_factor, end_factor=1.0, total_iters=args.warmup_steps)
            # Chain them together
            scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_steps])
            print_debug(rank_for_debug, f"LR Scheduler: SequentialLR created. Warmup steps={args.warmup_steps}, Decay steps={decay_steps}.")
        else:
            # No warmup, use main scheduler directly
            scheduler = main_scheduler
            print_debug(rank_for_debug, f"LR Scheduler: CosineAnnealingLR created directly. T_max={decay_steps} steps.")

    except Exception as e: logger.error(f"Error creating LR scheduler: {e}\n{traceback.format_exc()}"); cleanup(); sys.exit(1)
    # --- End LR Scheduler ---

    # --- Samplers ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed, drop_last=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp and has_val_data else None

    # --- DataLoaders ---
    try:
        # Use micro_batch_size for the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = None
        if has_val_data:
             # Validation batch size can be different, maybe larger if memory allows? Or keep same?
             val_batch_size = micro_batch_size # Keep same for simplicity
             val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        print_debug(rank_for_debug, f"Loaders created. Train micro-BS={micro_batch_size}. Val micro-BS={val_batch_size if has_val_data else 'N/A'}.")
    except Exception as e: logger.error(f"Error creating DataLoaders: {e}\n{traceback.format_exc()}"); cleanup(); sys.exit(1)

    # --- Training Prep ---
    update_ema(ema, model, decay=0) # Initialize EMA
    model.train()
    saved_batch_flag = False
    start_epoch = 0
    global_optim_step = 0 # Tracks optimizer steps completed by rank 0

    # --- Training Loop ---
    logger.info(f"Starting training from epoch {start_epoch} for up to {args.epochs} epochs...")
    training_should_continue = True
    epoch = 0 # Initialize loop var for final logging
    opt.zero_grad() # Ensure grads are zero at the beginning

    for epoch in range(start_epoch, args.epochs):
        if not training_should_continue: break

        epoch_start_time = time.time()
        model.train() # Set model to train mode at start of epoch
        if train_sampler: train_sampler.set_epoch(epoch)
        if rank == 0: logger.info(f"------ Beginning epoch {epoch} ------")

        running_train_loss = 0.0
        steps_in_epoch_logged = 0 # Track steps logged in current period
        micro_batches_processed_in_epoch = 0

        _batch_timer_start = time.time() # Timer for step duration logging

        # Iterate through micro-batches
        for i, batch_data in enumerate(train_loader):
            current_micro_batch_index = i + 1
            is_last_micro_batch_in_accum = (current_micro_batch_index % args.accumulation_steps == 0)

            if not batch_data: logger.warning(f"Skipping empty train micro-batch {current_micro_batch_index} in epoch {epoch}."); continue
            try: x, _ = batch_data
            except (TypeError, ValueError) as e: logger.error(f"Error unpacking micro-batch {current_micro_batch_index}: {e}"); continue

            try:
                x = x.to(device, non_blocking=True)
                # Save Batch Sample (Rank 0, First Actual Batch)
                if rank == 0 and not saved_batch_flag:
                    try:
                        save_path = "batch_sample_rank0.pt"; abs_save_path = os.path.abspath(save_path)
                        print_debug(rank, f"Saving batch sample (shape: {x.shape}) to {abs_save_path}...")
                        torch.save(x.cpu(), save_path); saved_batch_flag = True
                        print_debug(rank, f"Batch sample saved to {abs_save_path}.")
                    except Exception as save_e: logger.error(f"Failed to save batch sample: {save_e}")
            except Exception as e: logger.error(f"Error moving micro-batch {current_micro_batch_index} to device: {e}"); continue

            # --- Forward/Loss/Backward per micro-batch ---
            try:
                # If using DDP and gradient accumulation, need to handle model.no_sync()
                # Context manager to disable DDP gradient sync except on the last micro-batch
                sync_context = model.no_sync() if (is_ddp and not is_last_micro_batch_in_accum) else contextlib.nullcontext()

                with sync_context:
                    with torch.no_grad(): latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    t = torch.randint(0, diffusion.num_timesteps, (latent_x.shape[0],), device=device)
                    model_kwargs = dict()
                    loss_dict = diffusion.training_losses(model, latent_x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                    if not torch.isfinite(loss):
                        logger.error(f"Non-finite train loss at micro-batch idx {i}, epoch {epoch}: {loss.item()}. Skipping accum step.");
                        # Need to decide how to handle this. Skip the whole accum step?
                        # For now, just skip backward for this micro-batch.
                        loss = None # Signal to skip backward/accum
                    else:
                         # Scale loss for accumulation BEFORE backward
                         scaled_loss = loss / args.accumulation_steps
                         scaled_loss.backward() # Accumulate gradients

            except Exception as e:
                 logger.error(f"Model forward/loss/backward failed micro-batch idx {i}: {e}")
                 if "CUDA out of memory" in str(e): logger.error("OOM! Reduce micro-batch size or accumulation steps?"); cleanup(); sys.exit(1)
                 logger.error(traceback.format_exc()); loss = None # Skip accum

            # --- Accumulate Loss for Logging (using non-scaled loss) ---
            if loss is not None: # Only accumulate if forward/backward succeeded
                 loss_item_local = loss.item() # Use the original mean loss for logging average
                 running_train_loss += loss_item_local
                 steps_in_epoch_logged += 1
            # --- End Forward/Loss/Backward ---

            # --- Optimizer Step (Conditional) ---
            if is_last_micro_batch_in_accum:
                if rank == 0: print_debug(rank, f"Performing optimizer step after micro-batch index {i}.")
                try:
                    # Gradient Clipping (Applied to accumulated gradients)
                    if args.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                        # Log grad norm here if desired, associated with the global_optim_step about to be completed

                    opt.step()
                    scheduler.step() # Step scheduler after optimizer
                    opt.zero_grad() # Zero gradients *after* stepping

                    # --- Logging and Checkpointing (Rank 0 - Tied to Optimizer Step) ---
                    if rank == 0:
                        global_optim_step += 1 # Increment global optimizer step count

                        # Log Training Progress periodically based on optimizer steps
                        if global_optim_step % args.log_every == 0:
                            current_time = time.time(); step_time = current_time - _batch_timer_start; _batch_timer_start = current_time
                            # Average loss over micro-batches since last log
                            avg_loss_log = running_train_loss / steps_in_epoch_logged if steps_in_epoch_logged > 0 else 0
                            steps_per_sec = args.log_every / step_time if step_time > 0 else 0 # Steps/sec is optim steps / time
                            current_lr = scheduler.get_last_lr()[0] # Get current LR from scheduler
                            logger.info(f"Epoch {epoch} OptimStep {global_optim_step:07d}: TrainLoss={avg_loss_log:.4f} | LR={current_lr:.2e} | OptimStepTime={step_time:.2f}s ({steps_per_sec:.2f} optim steps/sec)")
                            # Reset averaging stats
                            running_train_loss = 0.0; steps_in_epoch_logged = 0

                        # Save Interval Checkpoint based on optimizer steps
                        if args.ckpt_every > 0 and global_optim_step % ckpt_every == 0:
                             if checkpoint_dir:
                                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{global_optim_step:07d}_epoch{epoch}.pt")
                                try:
                                     model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                                     checkpoint = {"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": epoch, "train_step": global_optim_step}
                                     torch.save(checkpoint, ckpt_path)
                                     logger.info(f"Saved interval checkpoint (Optim Step {global_optim_step}) to {ckpt_path}")
                                except Exception as e: logger.error(f"Error saving interval checkpoint: {e}")
                    # --- End Rank 0 Logging/Checkpointing ---

                except Exception as e: logger.error(f"Optimizer step/logging/checkpointing failed: {e}\n{traceback.format_exc()}");

            # Update EMA model weights after each micro-batch's backward pass
            update_ema(ema, model, decay=args.ema_decay)
            micro_batches_processed_in_epoch += 1
            # --- End Optimizer Step Conditional ---
        # --- End Micro-Batch Loop ---

        # --- Validation, Best Model Checkpointing, Early Stopping (End of Epoch) ---
        if val_loader is not None and epoch % args.val_every == 0:
            print_debug(rank_for_debug, f"Starting validation for epoch {epoch}.")
            avg_val_loss = validate_one_epoch(model, vae, diffusion, val_loader, device, rank, epoch, logger)
            print_debug(rank_for_debug, f"Finished validation for epoch {epoch}. Avg Loss: {avg_val_loss:.4f}")

            if rank == 0:
                logger.info(f"------ Epoch {epoch} Validation Summary ------")
                logger.info(f"Micro-batches processed this epoch: {micro_batches_processed_in_epoch}")
                logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
                # Check for improvement and save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss; best_val_epoch = epoch; epochs_no_improve = 0
                    logger.info(f"*** New best validation loss: {best_val_loss:.4f} at epoch {epoch} ***")
                    if checkpoint_dir: # Save best model based on validation loss
                        best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                        try:
                            model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                            best_checkpoint = {"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": epoch, "train_step": global_optim_step, "best_val_loss": best_val_loss}
                            torch.save(best_checkpoint, best_ckpt_path)
                            logger.info(f"Saved best model checkpoint (Val Loss: {best_val_loss:.4f}) to {best_ckpt_path}")
                        except Exception as e: logger.error(f"Error saving best checkpoint: {e}")
                else: # No improvement
                    if best_val_loss != float('inf'): epochs_no_improve += 1 # Only increment if we already had a best loss
                    logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best was {best_val_loss:.4f} at epoch {best_val_epoch}.")
                # Early Stopping Check
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    logger.warning(f"EARLY STOPPING triggered at epoch {epoch} after {args.patience} epochs without validation improvement.")
                    early_stop_triggered_local = True; training_should_continue = False
                logger.info(f"---------------------------------------")

        # --- Synchronize Early Stopping Signal (DDP) ---
        if is_ddp:
            stop_signal_tensor = torch.tensor(1.0 if early_stop_triggered_local else 0.0, device=device)
            dist.broadcast(stop_signal_tensor, src=0)
            if stop_signal_tensor.item() > 0.5: training_should_continue = False

        # --- Log Epoch Duration (Rank 0) ---
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"------ Epoch {epoch} Finished. Duration: {epoch_duration:.2f} seconds ------")
    # --- End Epoch Loop ---

    # --- Final Summary ---
    model.eval()
    if rank == 0:
        logger.info("="*40); logger.info("Training finished!")
        logger.info(f"Total optimizer steps completed: {global_optim_step}")
        logger.info(f"Best validation loss recorded: {best_val_loss:.4f} at epoch {best_val_epoch}")
        final_epoch_completed = epoch # 'epoch' holds the index of the last completed epoch
        if early_stop_triggered_local: logger.info(f"Training stopped early after completing epoch {final_epoch_completed}.")
        else: logger.info(f"Training completed planned {args.epochs} epochs (last epoch index: {final_epoch_completed}).")
        # Save final model state
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        if checkpoint_dir:
             logger.info(f"Saving final model state..."); model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
             last_val_loss = avg_val_loss if 'avg_val_loss' in locals() and has_val_data else float('inf')
             final_checkpoint = {"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": final_epoch_completed, "train_step": global_optim_step, "final_val_loss": last_val_loss}
             try: torch.save(final_checkpoint, final_checkpoint_path); logger.info(f"Saved final model state to {final_checkpoint_path}")
             except Exception as e: logger.error(f"Error saving final checkpoint: {e}")
        logger.info("="*40)

    # --- Cleanup ---
    if is_ddp: dist.barrier(); cleanup()
    print_debug(rank_for_debug, "Exiting main function.")

# --- Make sure if __name__ == "__main__": block follows and calls main(args) ---
if __name__ == "__main__":
    print("[DEBUG train.py __main__] Starting argument parsing...")
    parser = argparse.ArgumentParser()
    # +++ ADD/MODIFY ARGUMENTS +++
    parser.add_argument("--data-path", type=str, required=True, help="Path to the base data dir (containing 'train'/'val').")
    parser.add_argument("--results-dir", type=str, default="results_unconditional")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--attention-type", type=str, choices=['full', 'swa'], default='full')
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.") # Keep higher default
    parser.add_argument("--global-batch-size", type=int, default=64, help="Total effective batch size across all GPUs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay.")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N *optimizer* steps.")
    parser.add_argument("--ckpt-every", type=int, default=0, help="Save interval checkpoint every N *optimizer* steps (approx). 0 calculates dynamically.")
    parser.add_argument("--find-unused-parameters", action='store_true', default=False)
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs). 0 disables.")
    parser.add_argument("--scheduler-tmax-epochs", type=int, default=0, help="Epochs for scheduler T_max override.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max grad norm for clipping. 0 disables.")
    # +++ NEW ARGUMENTS +++
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of linear warmup *optimizer* steps. 0 disables warmup.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients over.")

    args = parser.parse_args()
    print("[DEBUG train.py __main__] Arguments parsed.")

    # --- Argument Validation ---
    if args.attention_type == 'swa' and (args.window_size <= 0 or args.window_size % 2 == 0): print(f"Error: --window-size must be positive odd for 'swa'."); sys.exit(1)
    if args.image_size % 8 != 0: print("Error: --image-size must be divisible by 8."); sys.exit(1)
    if args.patience < 0: print("Warning: --patience set to 0."); args.patience = 0
    if args.grad_clip_norm < 0: print("Warning: --grad-clip-norm set to 0."); args.grad_clip_norm = 0
    if args.warmup_steps < 0: print("Warning: --warmup-steps set to 0."); args.warmup_steps = 0
    if args.accumulation_steps < 1: print("Warning: --accumulation-steps set to 1."); args.accumulation_steps = 1
    # if args.global_batch_size % (world_size * args.accumulation_steps) != 0: # Check divisibility for micro-batch calc
    #      print(f"Error: Global batch size ({args.global_batch_size}) must be divisible by world_size*accumulation_steps ({world_size}*{args.accumulation_steps}).")
    #      # Or adjust logic to handle remainder batches, but simpler to enforce divisibility
    #      sys.exit(1)

    print("[DEBUG train.py __main__] Argument validation passed.")
    print("[DEBUG train.py __main__] Calling main function...")
    # Need to import contextlib for the model.no_sync() part
    import contextlib # Add this import near the top with others
    main(args)
    print("[DEBUG train.py __main__] main function returned.")

