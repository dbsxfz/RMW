import torch
from torch.nn import functional as F
import numpy as np


class SAIF(object):
    def __init__(self, model, epsilon=255 / 255, k=10, t=30, random_start=True, classes=10):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.t = t
        self.random_start = random_start
        self.classes = classes

    def update_perturbation(self, perturb, grad, x, t):
        alpha = 1 / np.sqrt(t + 1)
        z = self.epsilon * grad.sign()
        d = z - perturb
        perturb = perturb + alpha * d

        perturb = perturb.clamp_(-self.epsilon, self.epsilon)
        perturb = torch.min(torch.max(perturb, -x), 1 - x)
        return perturb

    def update_mask(self, mask, grad, t):
        b, c, h, w = mask.shape
        alpha = 1 / np.sqrt(t + 1)
        grad = grad.view(b, -1)
        _, idx = torch.sort(grad, dim=1, descending=True)
        # set the k largest elements of grad to 1, others to 0
        z = torch.zeros_like(grad).scatter_(1, idx[:, :self.k], 1).view(b, 1, h, w)
        d = z - mask
        mask = mask + alpha * d

        # sort the k largest elements in mask
        mask = mask.view(b, -1)
        _, idx = torch.sort(mask, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask = torch.zeros_like(mask).scatter_(1, idx[:, :self.k], 1).view(b, 1, h, w)
        return mask

    def initial_mask(self, x):
        b, c, h, w = x.size()
        initial_mask = torch.cat([torch.ones(self.k), torch.zeros(h * w - self.k)])
        mask = None

        for i in range(b):
            perm_idx = torch.randperm(h * w)
            mask_tmp = initial_mask[perm_idx].view(1, 1, h, w)
            mask = mask_tmp if mask is None else torch.cat([mask, mask_tmp], dim=0)
        mask = mask.to(x.device)
        return mask

    def initial_perturb(self, x, seed=-1):
        if self.random_start:
            if seed != -1:
                torch.random.manual_seed(seed)
            perturb = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new(x.size()).zero_()
        return perturb

    def __call__(self, x, y, seed=-1, targeted=False, target=None):
        if self.t == 0:
            return x
        perturb = self.initial_perturb(x, seed)
        # generate sparsity mask
        mask = self.initial_mask(x)
        mask_iter = 1

        training = self.model.training
        if training:
            self.model.eval()

        for i in range(self.t):
            perturb.requires_grad_()
            mask.requires_grad_()

            if targeted:
                loss = -F.cross_entropy(self.model(x + mask * perturb), target)
            else:
                loss = F.cross_entropy(self.model(x + mask * perturb), y)

            loss.backward()
            grad_perturb = perturb.grad.clone()
            grad_mask = mask.grad.clone()

            perturb = perturb.detach()
            mask = mask.detach()

            # update perturbation using Frank-Wolfe algorithm
            perturb = self.update_perturbation(perturb, grad_perturb, x, i)

            # update mask using Frank-Wolfe algorithm
            mask = self.update_mask(mask, grad_mask, i)
            mask_iter += 1

            with torch.no_grad():
                pred = self.model(x + mask * perturb).argmax(dim=1)
                acc = (pred == y).float()
                if torch.sum(acc) == 0.:
                    break

        if training:
            self.model.train()
        return perturb, mask, acc, pred, i
