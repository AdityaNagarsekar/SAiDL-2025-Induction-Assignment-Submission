"""Contains the core training and evaluation loops."""

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP # Only needed for isinstance check
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Need to import specific losses ONLY for the gradient clipping check
from losses import FocalLoss, NFLLoss

# === Training and Evaluation Functions ===
# Design Approach: Refined train/eval loops integrating AMP, gradient clipping,
# NaN handling, and optimized for the no-validation, last-epoch reporting methodology.

def train(model, train_loader, criterion, optimizer, device,
          scheduler=None, use_amp=True, scaler=None):
    """Performs one epoch of training with AMP, gradient clipping, and NaN checks."""
    model.train() # Set model to training mode (enables dropout if present, uses batch stats for BN)
    running_loss = 0.0
    correct = 0
    total = 0
    # Hyperparameter: Gradient clipping threshold (chosen heuristically)
    grad_clip_max_norm = 0.05

    # --- Determine if gradient clipping should be applied for this criterion ---
    apply_clipping = False
    loss_id = getattr(criterion, 'loss_name', type(criterion).__name__) # Get loss name/type for checking
    # Check if it's a standalone FL/NFL or an APL loss involving FL/NFL
    if isinstance(criterion, (FocalLoss, NFLLoss)) or ('FL' in loss_id): # Simple check using 'FL' substring
        apply_clipping = True
        # print(f"DEBUG: Applying gradient clipping for loss {loss_id}") # Optional debug print
    # --- End Clipping Check ---

    # Initialize progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        # Performance: Move data asynchronously if possible (requires pinned memory)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # Performance: Reset gradients efficiently
        optimizer.zero_grad(set_to_none=True)

        # --- Automatic Mixed Precision (AMP) Path ---
        # Condition: AMP enabled, GradScaler provided, and running on CUDA
        if use_amp and scaler and device.type == 'cuda':
            # AMP context manager: Operations inside run in lower precision (e.g., FP16)
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets) # Loss computed in mixed precision

            # Stability: Check for NaN loss *before* backpropagation
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected BEFORE backward (AMP) for {loss_id}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True) # Clear any partial gradients if needed
                continue # Skip this batch entirely

            # AMP: Scale the loss, then perform backward pass to calculate scaled gradients
            scaler.scale(loss).backward()

            # Stability: Apply gradient clipping if needed for this loss type
            if apply_clipping:
                # AMP: Unscale gradients before clipping to operate on original scale
                scaler.unscale_(optimizer)
                # Clip gradients in-place
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            # AMP: Optimizer step (automatically handles unscaling if not already done)
            scaler.step(optimizer)
            # AMP: Update the scale factor for the next iteration
            scaler.update()

        # --- Standard Precision Path (CPU or AMP disabled) ---
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Loss computed in FP32

            # Stability: Check for NaN loss *before* backpropagation
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected BEFORE backward (non-AMP) for {loss_id}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Standard backward pass
            loss.backward()

            # Stability: Apply gradient clipping if needed
            if apply_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            # Standard optimizer step
            optimizer.step()

        # --- Learning Rate Scheduler Step (Batch-wise) ---
        # Design Choice: Support schedulers that update per batch (e.g., OneCycleLR)
        if scheduler is not None and isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            scheduler.step()
        # --- End Scheduler Step ---

        # --- Track Statistics ---
        batch_loss = loss.item() # Get Python float value of the loss
        running_loss += batch_loss * inputs.size(0) # Accumulate loss, weighted by batch size
        _, predicted = outputs.max(1) # Get predicted class index
        total += targets.size(0) # Accumulate total samples processed
        correct += predicted.eq(targets).sum().item() # Accumulate correct predictions
        # Update progress bar display
        pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
        # --- End Statistics ---

    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / total if total > 0 else float('nan') # Avoid division by zero
    train_acc = 100.0 * correct / total if total > 0 else 0.0
    return train_loss, train_acc


def evaluate(model, data_loader, criterion, device, use_amp=True, desc="Evaluating"):
    """Evaluates the model on the given dataloader, calculating loss and accuracy.
       Uses AMP if enabled. Handles optional criterion. Returns ACCURACY.
    """
    model.eval() # Set model to evaluation mode (disables dropout, BN uses running stats)
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=desc, leave=False) # Progress bar
    # Disable gradient computations for efficiency and correctness during evaluation
    with torch.no_grad():
        for inputs, targets in pbar:
            # Performance: Move data asynchronously
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Determine appropriate device type for autocast (supports 'cuda', 'mps')
            amp_device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'

            # --- AMP Path for Evaluation ---
            if use_amp and amp_device_type != 'cpu':
                with autocast(device_type=amp_device_type):
                    outputs = model(inputs)
                    # Calculate loss only if a criterion is provided
                    if criterion:
                        loss = criterion(outputs, targets)
                    else:
                        loss = torch.tensor(0.0, device=device) # Placeholder if no loss needed
            # --- Standard Precision Path ---
            else:
                outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)
                else:
                    loss = torch.tensor(0.0, device=device)

            # --- Track Statistics ---
            batch_loss = loss.item() if criterion else 0.0 # Get loss value if calculated
            running_loss += batch_loss * inputs.size(0) # Accumulate loss
            _, predicted = outputs.max(1) # Get predictions
            total += targets.size(0) # Count samples
            correct += predicted.eq(targets).sum().item() # Count correct predictions
            # Update progress bar - NOTE: Reports Accuracy!
            pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
            # --- End Statistics ---

    # Calculate overall epoch results
    eval_loss = running_loss / total if total > 0 else 0.0 # Avoid division by zero
    # Design Choice / Correction (Iter 6): Calculate and return ACCURACY
    eval_acc = 100.0 * correct / total if total > 0 else 0.0
    return eval_loss, eval_acc