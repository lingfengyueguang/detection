# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:30:23 2019

@author: goodxin
"""

import torch.nn.functional as F



"""
   This is a Pytorch implementation of Focal loss.
    Args:
        pred: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): Weights of various categories
    
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 

        In the paper, focal loss should be normalized by the number ofanchors assigned to a ground-truth box.
    It means focal loss should be normalized by the number of positive  samples,so in this function we give a 
    sum loss. You should divide the loss by num of pos out this function.
    
    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss = loss / num_pos
"""

def focal_loss(pred,
                   target,
                   weight=None,
                   gamma=2.0,
                   alpha=0.25,
                   reduction='sum'):
    
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    if weight is None :
        weight = (alpha * target + (1 - alpha) * (1 - target))
    else:
       weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target,weight=weight,reduction =reduction)
    return loss
