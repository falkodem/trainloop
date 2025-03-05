import torch

def one_hot_labels(labels: torch.Tensor, n_classes, dim=1):
    '''
    Params:
        labels (torch.Tensor): BxCxHxW, where C - channels and should be equal to 1
    '''
    sh = list(labels.shape)
    if sh[dim] != 1:
        raise AssertionError(f'Labels dim (for that case dim{dim}) should be equal to 1 (now is equal to {sh[dim]})')
    sh[dim] = n_classes
    return torch.zeros(size=sh, dtype=labels.dtype, device=labels.device).scatter_(dim=dim, index=labels, value=1)