"""
OPTIMUS Learning Rate Scheduler
================================
Linear Warmup + Cosine Annealing to minimum learning rate.

Schedule:
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + cos(π * decay_ratio))
        lr = min_lr + coeff * (max_lr - min_lr)
"""

import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr_ratio`.
    
    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps for linear ramp.
        total_steps: Total number of training steps (including warmup).
        min_lr_ratio: Ratio of min_lr / max_lr. The cosine decays to this fraction.
        last_epoch: The index of the last epoch (for resuming).
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # LambdaLR calls lr_lambda immediately, so set attributes before super().__init__
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step):
        """
        Returns a multiplicative factor for the base learning rate.
        
        - During warmup: linearly ramp from 0 to 1
        - After warmup: cosine anneal from 1 to min_lr_ratio
        """
        if step < self.warmup_steps:
            # Linear warmup: 0 → 1
            return float(step) / float(max(1, self.warmup_steps))
        
        if step >= self.total_steps:
            # After all steps, stay at minimum
            return self.min_lr_ratio
        
        # Cosine annealing: 1 → min_lr_ratio
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + cosine_coeff * (1.0 - self.min_lr_ratio)


def build_scheduler(optimizer, training_config, total_steps):
    """
    Factory function to build the WarmupCosineScheduler.
    
    Args:
        optimizer: The optimizer to schedule.
        training_config: TrainingConfig instance.
        total_steps: Total optimizer steps across all epochs.
    
    Returns:
        WarmupCosineScheduler instance.
    """
    min_lr_ratio = training_config.min_lr / training_config.learning_rate
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=training_config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
