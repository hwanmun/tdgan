""" Loss functions for generator and discriminator. """

import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logit

def gen_BCE_logit_loss(fake_logit):
    return bce_logit(fake_logit, torch.ones_like(fake_logit))

def disc_BCE_logit_loss(real_logit, fake_logit):
    real_loss = bce_logit(real_logit, torch.ones_like(fake_logit))
    fake_loss = bce_logit(fake_logit, torch.zeros_like(fake_logit))
    return (real_loss + fake_loss) / 2
