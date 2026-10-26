"""Research-only masked mean for a future representation experiment.

No model loading or school-scanner integration. Correcting query pooling also
requires regenerating the reference index with the same representation identity.
"""
import torch


def masked_mean(hidden, attention_mask):
    """Pool BxTxD or LxBxTxD states into BxD or BxLxD, without mutation.

    Masks must be binary BxT tensors on the same device. Every row must contain
    a valid token. Nonfinite valid states are rejected; masked states are ignored.
    This function does not normalize embeddings or choose a classification rule.
    """
    if not isinstance(hidden, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise ValueError('Hidden states and mask must be tensors.')
    if hidden.ndim not in (3, 4) or attention_mask.ndim != 2:
        raise ValueError('Expected BxTxD or LxBxTxD states and a BxT mask.')
    if not hidden.is_floating_point() or attention_mask.is_complex():
        raise ValueError('Hidden states must be floating point and masks real binary values.')
    if any(size == 0 for size in hidden.shape):
        raise ValueError('Empty batch, layer, token or feature dimension.')
    if tuple(hidden.shape[-3:-1]) != tuple(attention_mask.shape):
        raise ValueError('Mask shape does not match batch and token dimensions.')
    if hidden.device != attention_mask.device:
        raise ValueError('Hidden states and mask must use the same device.')
    if not torch.all((attention_mask == 0) | (attention_mask == 1)).item():
        raise ValueError('Attention mask must contain only zero or one.')
    valid = attention_mask.bool()
    counts = valid.sum(dim=1)
    if torch.any(counts == 0).item():
        raise ValueError('Every example must have at least one valid token.')
    mask = valid[..., None] if hidden.ndim == 3 else valid[None, ..., None]
    # masked_fill returns a new tensor; discarding it leaves padding in the sum.
    masked = hidden.masked_fill(~mask, 0.0)
    if not torch.isfinite(masked).all().item():
        raise ValueError('Nonfinite valid hidden state.')
    if hidden.ndim == 3:
        pooled = masked.sum(dim=1) / counts[:, None]
    else:
        pooled = (masked.sum(dim=2) / counts[None, :, None]).permute(1, 0, 2)
    if not torch.isfinite(pooled).all().item():
        raise ValueError('Nonfinite pooled output.')
    return pooled
