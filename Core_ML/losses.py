# losses.py
"""Contains implementations of various loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === Common Helper Function ===
def normalize_loss(numerator, denominator, epsilon=1e-7):
    """
    Applies the core normalization logic: Numerator / (Denominator + epsilon).
    Used by normalized loss functions (NCE, NFL, NMAE, NRCE).

    Args:
        numerator (Tensor): The loss term for the target class (or equivalent).
        denominator (Tensor): The sum of loss terms over all classes (or equivalent).
        epsilon (float): Small value for numerical stability to prevent division by zero.

    Returns:
        Tensor: The normalized loss value(s).
    """
    # Ensure stability by adding epsilon to the denominator before division
    return numerator / (denominator + epsilon)


# === Loss Function Implementations ===
# prioritizing stability (esp. FL/NFL) and alignment with paper definitions/parameters.

# --- Standard Losses (Baseline, Active) ---
class CrossEntropyLoss(nn.Module):
    """Standard Cross Entropy Loss wrapper."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        """Calculates CE loss using PyTorch's functional implementation."""
        return F.cross_entropy(logits, targets)

class FocalLoss(nn.Module):
    """Focal Loss (FL) implementation with numerical stability enhancements. Active Loss.
       Uses gamma=0.5 default, aligning with Ma et al. CIFAR experiments.
    """
    def __init__(self, gamma=0.5, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        # gamma: Focusing parameter. Higher values focus more on hard examples.
        # reduction: Specifies aggregation method ('mean', 'sum', 'none').
        # epsilon: Small value for numerical stability in intermediate calculations.
        # prob_clip: Value to clip probabilities away from exact 0 or 1.
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.prob_clip = 1e-6 # Stability: Avoid log(0) or (1-p)**gamma when p=1

    def forward(self, logits, targets):
        # Stability: Use log_softmax for better numerical precision than softmax -> log.
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Get the log-probabilities corresponding to the target classes. Shape: [N]
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Use squeeze(1)

        # Calculate Cross Entropy loss per sample in a stable way: CE = -log(p_target)
        ce_loss = -target_log_probs # Shape: [N]

        # Calculate probabilities (pt) for the target classes, ensuring stability.
        pt = torch.exp(target_log_probs) # Shape: [N]
        # Stability: Clamp probabilities to avoid issues at the boundaries (0 or 1).
        pt = torch.clamp(pt, min=self.prob_clip, max=1.0 - self.prob_clip)

        # Calculate the focal loss modulating factor: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma # Shape: [N]
        # Stability: Clip focal weight to prevent potential explosion if pt is tiny and gamma<0 (though gamma>=0 here)
        # or just generally keep weights reasonable. Max value chosen empirically/heuristically.
        focal_weight = torch.clamp(focal_weight, min=0.0, max=1e3)

        # Compute the final Focal Loss per sample: weight * CE_loss
        focal_loss = focal_weight * ce_loss # Shape: [N]

        # Stability: Final safety check for NaNs. If NaN occurs (should be rare with fixes), fallback to CE.
        if torch.isnan(focal_loss).any():
            print(f"Warning: NaN detected in FocalLoss (gamma={self.gamma}). Falling back to CE loss for this batch.")
            focal_loss = ce_loss # Use the pre-calculated stable CE loss

        # Apply reduction based on the specified mode.
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


# --- Baseline Robust Losses (Passive) ---
class MAELoss(nn.Module):
    """Mean Absolute Error (MAE) Loss. Known to be robust to noise. Passive Loss.
       Calculates Sum_k |p_k - q_k|, where q_k is one-hot target.
    """
    def __init__(self, num_classes=10, reduction='mean'):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        # Calculate predicted probabilities
        probs = F.softmax(logits, dim=1) # Shape: [N, C]
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, self.num_classes).float().to(device) # Shape: [N, C]
        # Calculate MAE per sample: sum of absolute differences across classes
        mae_per_sample = torch.abs(probs - targets_one_hot).sum(dim=1) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return mae_per_sample.mean()
        elif self.reduction == 'sum':
            return mae_per_sample.sum()
        else: # 'none'
            return mae_per_sample

class RCELoss(nn.Module):
    """Reverse Cross Entropy (RCE) Loss. Known to be robust. Passive Loss.
       Uses the simplified form derived from definition: RCE = -A * (1 - p_y).
       Uses A=-4.0 default, aligning with Ma et al. CIFAR experiments.
    """
    def __init__(self, num_classes=10, A=-4.0, reduction='mean'):
        super(RCELoss, self).__init__()
        if A >= 0: raise ValueError("RCE parameter 'A' must be negative.")
        self.num_classes = num_classes
        self.A = A # Log-value for pseudo-targets of non-labeled classes
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1) # Shape: [N, C]
        # Get probabilities corresponding to the target classes
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]

        # Calculate RCE per sample using the simplified formula
        # Derived from -Sum[p_k * log(q_k)] where log(q_y)=log(1)=0 and log(q_{k!=y})=A
        # -> - [ p_y*log(q_y) + Sum_{k!=y} p_k*log(q_k) ]
        # -> - [ 0 + Sum_{k!=y} p_k*A ] = -A * Sum_{k!=y} p_k = -A * (1 - p_y)
        rce_per_sample = -self.A * (1.0 - p_y) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return rce_per_sample.mean()
        elif self.reduction == 'sum':
            return rce_per_sample.sum()
        else: # 'none'
            return rce_per_sample


# --- Normalized Losses (Active or Passive depending on base loss) ---
class NCELoss(nn.Module):
    """Normalized Cross Entropy (NCE) Loss. Active Loss.
       Implements L_norm = L_CE(y) / Sum_j L_CE(j) = (-log p_y) / (-Sum_j log p_j).
    """
    def __init__(self, epsilon=1e-7, reduction='mean'):
        super(NCELoss, self).__init__()
        self.epsilon = epsilon # For numerical stability in division
        self.reduction = reduction

    def forward(self, logits, targets):
        # Use log_softmax for stability
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Get log-probability of the target class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]

        # Numerator for normalization: CE loss for the target class
        numerator = -target_log_probs # Shape: [N]
        # Denominator for normalization: Sum of CE losses over all classes
        denominator = -torch.sum(log_probs, dim=1) # Shape: [N]

        # Apply normalization using the helper function
        nce_loss_per_sample = normalize_loss(numerator, denominator, self.epsilon) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nce_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return nce_loss_per_sample.sum()
        else: # 'none'
            return nce_loss_per_sample

class NFLLoss(nn.Module):
    """Normalized Focal Loss (NFL). Active Loss.
       Implements L_norm = L_FL(y) / Sum_j L_FL(j).
       Uses gamma=0.5 default. Includes extensive stability enhancements.
    """
    def __init__(self, gamma=0.5, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon # Epsilon for the final normalization division
        self.reduction = reduction
        self.prob_clip = 1e-6 # Probability clipping value

    def forward(self, logits, targets):
        # Stability: Use log_softmax
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Stability: Calculate probabilities and clip them
        probs = torch.exp(log_probs)
        probs = torch.clamp(probs, min=self.prob_clip, max=1.0 - self.prob_clip)
        # Stability: Recompute log_probs from clipped probs for consistency
        log_probs = torch.log(probs)

        # Calculate the focal loss term -(1-p_k)^gamma * log(p_k) for ALL classes
        focal_term_all_classes = -( (1.0 - probs) ** self.gamma ) * log_probs # Shape: [N, C]

        # Numerator: Focal loss term for the target class y
        fl_y = focal_term_all_classes.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]
        # Denominator: Sum of focal loss terms over all classes j
        fl_all_sum = torch.sum(focal_term_all_classes, dim=1) # Shape: [N]

        # Stability Check: Warn if denominator is extremely small, indicating potential instability.
        if (fl_all_sum.abs() < self.epsilon).any():
             print(f"Warning: NFLLoss denominator near zero (min abs={fl_all_sum.abs().min()}).")

        # Apply normalization
        nfl_loss_per_sample = normalize_loss(fl_y, fl_all_sum, self.epsilon) # Shape: [N]

        # Stability Check: Final check for NaNs AFTER normalization. Raise error if found.
        if torch.isnan(nfl_loss_per_sample).any():
            # Provide more context if possible (consider printing intermediate values here if debugging)
            raise ValueError("NaN detected in NFLLoss calculation AFTER normalization. Training cannot proceed.")

        # Apply reduction
        if self.reduction == 'mean':
            return nfl_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return nfl_loss_per_sample.sum()
        else: # 'none'
            return nfl_loss_per_sample


class NMAELoss(nn.Module):
    """Normalized Mean Absolute Error (NMAE) Loss. Passive Loss.
       Uses the simplified analytical form: NMAE = MAE / (2*(K-1)).
    """
    def __init__(self, num_classes=10, reduction='mean'):
        super(NMAELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        # Pre-compute the scaling factor denominator for efficiency
        self.scale_factor = 2.0 * (self.num_classes - 1.0) # Use float
        # Safety check for K=1 case (though unlikely)
        if abs(self.scale_factor) < 1e-9:
            print("Warning: NMAE scale factor near zero (K=1?). Setting scale factor to 1.")
            self.scale_factor = 1.0

    def forward(self, logits, targets):
        device = logits.device
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float().to(device)
        # Calculate MAE per sample first
        mae_per_sample = torch.abs(probs - targets_one_hot).sum(dim=1) # Shape: [N]
        # Apply the pre-computed scaling factor
        nmae_per_sample = mae_per_sample / self.scale_factor # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nmae_per_sample.mean()
        elif self.reduction == 'sum':
            return nmae_per_sample.sum()
        else: # 'none'
            return nmae_per_sample

class NRCELoss(nn.Module):
    """Normalized Reverse Cross Entropy (NRCE) Loss. Passive Loss.
       Uses the simplified analytical form: NRCE = RCE / |A*(K-1)| = (1-p_y)/(K-1) (for A<0).
    """
    def __init__(self, num_classes=10, A=-4.0, reduction='mean'):
        super(NRCELoss, self).__init__()
        if A >= 0: raise ValueError("NRCE parameter 'A' must be negative.")
        self.num_classes = num_classes
        self.A = A
        self.reduction = reduction
        # Pre-compute the scaling factor denominator: |A * (K-1)|
        self.scale_denominator = abs(self.A * (self.num_classes - 1.0)) # Use float
        # Safety check
        if abs(self.scale_denominator) < 1e-9:
             print("Warning: NRCE scale denominator near zero (K=1 or A=0?). Setting denominator to 1.")
             self.scale_denominator = 1.0
        # Design Choice: Instantiate RCE internally to get per-sample values easily.
        # Could also recalculate RCE here, but this reuses code.
        self.rce_internal = RCELoss(num_classes=self.num_classes, A=self.A, reduction='none')

    def forward(self, logits, targets):
        # Calculate RCE per sample using the internal instance
        rce_per_sample = self.rce_internal(logits, targets) # Shape: [N]
        # Apply the pre-computed scaling factor denominator
        nrce_per_sample = rce_per_sample / self.scale_denominator # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nrce_per_sample.mean()
        elif self.reduction == 'sum':
            return nrce_per_sample.sum()
        else: # 'none'
            return nrce_per_sample


# --- Active Passive Loss (APL) Framework ---
class APLLoss(nn.Module):
    """Implements the Active Passive Loss framework: alpha * Active + beta * Passive.
       Requires constituent active and passive losses to be robust (inherently or normalized).
    """
    def __init__(self, active_loss, passive_loss, alpha=1.0, beta=1.0, loss_name=None):
        super(APLLoss, self).__init__()
        self.active_loss = active_loss   # The robust active loss component (e.g., NCE, NFL instance)
        self.passive_loss = passive_loss # The robust passive loss component (e.g., MAE, RCE instance)
        self.alpha = alpha               # Weight for the active term
        self.beta = beta                 # Weight for the passive term
        self.loss_name = loss_name       # String name for identification (e.g., "NCE+MAE")

    def forward(self, logits, targets):
        """Calculates the combined APL loss."""
        # Calculate the active component loss
        active_term = self.active_loss(logits, targets)
        # Calculate the passive component loss
        passive_term = self.passive_loss(logits, targets)
        # Return the weighted sum
        return self.alpha * active_term + self.beta * passive_term