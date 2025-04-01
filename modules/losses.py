import torch
import torch.nn.functional as F

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    assert avg_factor is not None, 'avg_factor can not be None'

    # if reduction is mean, then average the loss by avg_factor
    if reduction == 'mean':
        loss = loss.sum() / avg_factor
    # if reduction is 'none', then do nothing, otherwise raise an error
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean',
                          avg_factor=None, loss_weight=1.):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred).unsqueeze(1)

    num_classes = pred_sigmoid.shape[1]
    class_range = torch.arange(1, num_classes + 1, dtype=pred_sigmoid.dtype, device='cuda').unsqueeze(0)
    target = (target == class_range).float()

    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss_weight * loss