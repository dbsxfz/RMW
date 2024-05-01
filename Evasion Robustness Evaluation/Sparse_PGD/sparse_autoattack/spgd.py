import torch
from torch.nn import functional as F
import numpy as np


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


def dlr_loss(x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def margin_loss(logits, x, y, targeted=False):
        """
        :param y:        correct labels if untargeted else target labels
        """
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if not targeted:
            return y_corr - y_others
        else:
            return y_others - y_corr


class MaskingA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        # mask_back = mask.clone().view(b, -1)
        # mask_back = mask_back.scatter_(1, idx[:, k:], 0).view(b, 1, h, w)
        # ctx.save_for_backward(x, mask_back)

        ctx.save_for_backward(x, mask)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None


class MaskingB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask_proj)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None
        



class SparsePGD(object):
    def __init__(self, model, epsilon=255 / 255, k=10, t=30, random_start=True, patience=3, classes=10, alpha=0.25,
                 beta=0.25, trigger=None, unprojected_gradient=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.t = t
        self.random_start = random_start
        self.alpha = epsilon * alpha
        self.beta = beta
        self.patience = patience
        self.classes = classes
        self.masking = MaskingA() if unprojected_gradient else MaskingB()
        self.weight_decay = 0.0
        self.trigger = trigger

    def initial_perturb(self, x, seed=-1):
        if self.trigger is not None:
            perturb = self.trigger + x.new(x.size()).uniform_(-self.epsilon / 16, self.epsilon / 16)
        elif self.random_start:
            if seed != -1:
                torch.random.manual_seed(seed)
            perturb = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new(x.size()).zero_()
        perturb = torch.min(torch.max(perturb, -x), 1 - x)
        return perturb

    def update_perturbation(self, perturb, grad, perturb_old, x, it):
        perturb1 = perturb + self.alpha * grad.sign()
        perturb1 = perturb1.clamp_(-self.epsilon, self.epsilon)
        perturb1 = torch.min(torch.max(perturb1, -x), 1 - x)
        return perturb1, perturb_old


    def update_mask(self, mask, grad, it):

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()

        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        d = grad / (grad_norm + 1e-10)

        step_size = np.sqrt(h*w*c) * self.beta * torch.ones(b, device=mask.device)
        step_size = step_size.scatter_(0, (grad_norm.view(-1) < 2e-10).nonzero().squeeze(), 0)
        mask = mask + step_size.view(b, 1, 1, 1) * d

        # soft thresholding
        # idx = torch.where(mask != 0)
        #
        # mask[idx] = torch.sign(mask[idx]) * torch.maximum(
        #     torch.abs(mask[idx]) - self.beta * np.sqrt(h * w * c) * self.weight_decay,
        #     torch.zeros_like(mask[idx]))
        # print(mask.max(), mask.min(), mask.mean())
        return mask, grad.clone()

    ###############################################
    def initial_mask(self, x, it=0, prev_mask=None):

        if x.dim() == 3:
            x = x.unsqueeze(0)
        b, c, h, w = x.size()
        # random initial partial mask
        trigger_single_channel = self.trigger.abs().max(dim=0, keepdim=True)[0]
        mask = (trigger_single_channel > 1/255).float()
        mask = mask.expand(b, 1, h, w).to(x.device) + (torch.randn(b, 1, h, w) / 16).to(x.device)
        # mask = torch.randn(b, 1, h, w).to(x.device)
        # mask = np.sqrt(3) * (2 * torch.rand(b, 1, h, w).to(x.device) - 1)
        return mask

    def project_mask(self, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :self.k], 1).view(b, c, h, w)
        return mask_proj

    def __call__(self, x, y, seed=-1, targeted=False, target=None):
        if self.t == 0:
            return x
        b, c, h, w = x.size()
        perturb = self.initial_perturb(x, seed)

        perturb_old = perturb.clone()
        # generate sparsity mask
        mask = self.initial_mask(x)

        training = self.model.training
        if training:
            self.model.eval()

        reinitial_count = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mask_iter = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        perturb_best = perturb.clone()
        mask_best = mask.clone()


        # First loop
        perturb.requires_grad_()
        mask.requires_grad_()
        proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
        with torch.no_grad():
            assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
            assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(torch.min(x + proj_perturb).item(),
            torch.max(x + proj_perturb).item())
        logits = self.model(x + proj_perturb)

        if targeted:
            # loss = dlr_loss_targeted(logits, y, target)
            loss = -F.cross_entropy(logits, target, reduction='none')
        else:
            loss = F.cross_entropy(logits, y, reduction='none')
            # loss = dlr_loss(logits, y)

        loss.sum().backward()
        grad_perturb = perturb.grad.clone()
        grad_mask = mask.grad.clone()

        loss_best = loss.detach().clone()

        # prev_mask = mask.clone()
        mask_grad_old = torch.zeros_like(mask)

        for i in range(self.t):
            perturb = perturb.detach()
            mask = mask.detach()

            # update mask
            # prev_mask = 0.25 * prev_mask + 0.75 * mask.clone()
            prev_mask = mask.clone()

            mask, mask_grad_old = self.update_mask(mask, grad_mask, mask_iter)

            mask_iter = mask_iter + 1

            # update perturbation using PGD
            perturb, perturb_old = self.update_perturbation(perturb=perturb, grad=grad_perturb, perturb_old=perturb_old,
                                                            x=x, it=i)

            # forward pass
            perturb.requires_grad_()
            mask.requires_grad_()
            proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
            # proj_perturb = self.masking.apply(perturb, mask, self.k)
            with torch.no_grad():
                assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
                assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(torch.min(x + proj_perturb).item(),
            torch.max(x + proj_perturb).item())
            logits = self.model(x + proj_perturb)

            if targeted:
                # loss = dlr_loss_targeted(logits, y, target)
                loss = -F.cross_entropy(logits, target, reduction='none')
            else:
                # loss = dlr_loss(logits, y)
                loss = F.cross_entropy(logits, y, reduction='none')

            # backward pass
            loss.sum().backward()
            grad_perturb = perturb.grad.clone()
            grad_mask = mask.grad.clone()

            # save the best perturbation and mask, and check mask
            with torch.no_grad():
                logits = logits.detach()
                loss = loss.detach()
                # print(F.cross_entropy(logits, y, reduction='mean'), dlr_loss(logits, y).mean())

                fool_label = logits.argmax(dim=1)
                acc = (fool_label == y).float()

                ind_loss_improve = (loss >= loss_best).nonzero().squeeze()

                loss_best[ind_loss_improve] = loss[ind_loss_improve].clone()
                perturb_best[ind_loss_improve] = perturb[ind_loss_improve].clone()
                mask_best[ind_loss_improve] = mask[ind_loss_improve].clone()

                # print('cosine similarity:', torch.cosine_similarity(grad_mask[ind_loss_improve].view(-1, h*w),
                #                                                     mask_grad_old[ind_loss_improve].view(-1, h*w),
                #                                                     dim=1).mean().item())

                # ind_black_loss_improve = ((black_loss >= loss_best) * (black_loss >= loss)).nonzero().squeeze()
                # mask_best[ind_black_loss_improve] = black_mask[ind_black_loss_improve].clone()

                ind_fail = ((acc == 1)).nonzero().squeeze()
                # ind_fail = ((acc == 1) * (loss >= loss_best)).nonzero().squeeze()
                if ind_fail.numel() > 0:
                    # print(i, torch.max(mask[ind_fail]), torch.min(mask[ind_fail]), torch.median(mask[ind_fail]))
                    delta_mask_norm = torch.norm(self.project_mask(mask[ind_fail]) - self.project_mask(prev_mask[ind_fail]),
                                                 p=0, dim=(1, 2, 3))
                    ind_count = (delta_mask_norm <= 0).nonzero().squeeze()
                    if ind_count.numel() > 0:
                        if ind_fail.numel() == 1:
                            reinitial_count[ind_fail] += 1
                            # weight_decay[ind_fail] /= 2
                        else:
                            reinitial_count[ind_fail[ind_count]] += 1
                            # weight_decay[ind_fail[ind_count]] /= 2
                    else:
                        reinitial_count[ind_fail] = 0
                        # weight_decay[ind_fail] = self.weight_decay

                    ind_reinit = (reinitial_count >= self.patience).nonzero().squeeze()
                    if ind_reinit.numel() > 0:
                        mask[ind_reinit] = self.initial_mask(x[ind_reinit])
                        prev_mask[ind_reinit] = mask[ind_reinit].clone()
                        perturb[ind_reinit] = self.initial_perturb(x[ind_reinit])
                        reinitial_count[ind_reinit] = 0
                        mask_iter[ind_reinit] = 0

                        # weight_decay[ind_reinit] = self.weight_decay
                        # print('Triggered reinitialization at {}-th iteration, {} samples'.format(i, ind_reinit.numel()))

                if torch.sum(acc) == 0.:
                    break

        if training:
            self.model.train()
        return perturb_best, mask_best, acc, i

    def perturb(self, x, y):
        perturb, mask, acc, i = self.__call__(x, y, targeted=False)
        return x + self.masking.apply(perturb, F.sigmoid(mask), self.k), F.sigmoid(mask)

    def ensemble_attack(self, x, y):
        original_y = y.clone()
        # clean results
        clean_label = self.model(x).argmax(dim=1)
        fool_label = clean_label.clone()
        clean_acc = (clean_label == y).float().sum().item()
        ind_fail = (clean_label == y).nonzero().squeeze()
        x, y = x[ind_fail], y[ind_fail]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # untargeted attack
        if ind_fail.numel() > 0:
            perturb, mask, acc, i = self.__call__(x, y, targeted=False)
            pred = self.model(x + self.masking.apply(perturb, F.sigmoid(mask), self.k)).argmax(dim=1)
            # pred = self.model(x + self.masking.apply(perturb, mask, self.k)).argmax(dim=1)
            ind_untargeted_success = (pred != y).nonzero().squeeze()
            if ind_untargeted_success.numel() > 0:
                if ind_fail.numel() == 1:
                    fool_label[ind_fail] = pred[ind_untargeted_success]
                else:
                    fool_label[ind_fail[ind_untargeted_success]] = pred[ind_untargeted_success]
            ind_untargeted_fail = (pred == y).nonzero().squeeze()
            x, y = x[ind_untargeted_fail], y[ind_untargeted_fail]
            if x.dim() == 3:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
            ind_fail = ind_fail[ind_untargeted_fail]
            
            robust_acc_ut = (fool_label == original_y).float().sum().item()

            # targeted attack
            if ind_fail.numel() > 0:
                class_candidates = torch.arange(self.classes).expand(x.size(0), self.classes).to(y.device)
                class_candidates = class_candidates[(class_candidates != y.unsqueeze(1))].view(x.size(0), -1).permute(1, 0)
                ind_fail2 = torch.arange(x.size(0)).to(x.device)
                for i in range(class_candidates.size(0)):
                    target_class = class_candidates[i, ind_fail2]
                    if target_class.dim() == 0:
                        target_class = target_class.unsqueeze(0)
                    perturb, mask, acc, i = self.__call__(x, y, targeted=True, target=target_class)
                    pred = self.model(x + self.masking.apply(perturb, F.sigmoid(mask), self.k)).argmax(dim=1)
                    # pred = self.model(x + self.masking.apply(perturb, mask, self.k)).argmax(dim=1)
                    ind_targeted_success = (pred == target_class).nonzero().squeeze()
                    if ind_targeted_success.numel() > 0:
                        if ind_fail.numel() == 1:
                            fool_label[ind_fail] = pred[ind_targeted_success]
                        else:
                            fool_label[ind_fail[ind_targeted_success]] = pred[ind_targeted_success]

                    ind_targeted_fail = (pred != target_class).nonzero().squeeze()
                    ind_fail = ind_fail[ind_targeted_fail] if ind_fail.numel() > 1 else ind_fail
                    ind_fail2 = ind_fail2[ind_targeted_fail] if ind_fail2.numel() > 1 else ind_fail2
                    if ind_targeted_fail.numel() == 0:
                        break
                    x, y = x[ind_targeted_fail], y[ind_targeted_fail]
                    if x.dim() == 3:
                        x = x.unsqueeze(0)
                        y = y.unsqueeze(0)

        robust_acc = (fool_label == original_y).float().sum().item()
        return clean_acc, robust_acc, robust_acc_ut, ind_fail

    def change_masking(self):
        if isinstance(self.masking, MaskingA):
            self.masking = MaskingB()
        else:
            self.masking = MaskingA()



