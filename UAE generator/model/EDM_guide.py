import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


class EDM_guide(nn.Module):
    def __init__(self, nn_model,
                 sigma_data, p_mean, p_std,
                 sigma_min, sigma_max, rho,
                 S_min, S_max, S_noise,
                 device, drop_prob=0.1):
        ''' EDM proposed by "Elucidating the Design Space of Diffusion-Based Generative Models".

            Args:
                nn_model: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Training parameters:
                sigma_data, p_mean, p_std
                drop_prob
            Sampling parameters:
                sigma_min, sigma_max, rho
                S_min, S_max, S_noise
        '''
        super(EDM_guide, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device
        self.drop_prob = drop_prob

        def number_to_torch_device(value):
            return torch.tensor(value).to(device)

        self.sigma_data = number_to_torch_device(sigma_data)
        self.p_mean     = number_to_torch_device(p_mean)
        self.p_std      = number_to_torch_device(p_std)
        self.sigma_min  = number_to_torch_device(sigma_min)
        self.sigma_max  = number_to_torch_device(sigma_max)
        self.rho        = number_to_torch_device(rho)
        self.S_min      = number_to_torch_device(S_min)
        self.S_max      = number_to_torch_device(S_max)
        self.S_noise    = number_to_torch_device(S_noise)

    def perturb(self, x, t=None, steps=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, steps]`. Type: int / torch.LongTensor / None. \
                    Random `ln(sigma) ~ N(P_mean, P_std)` is taken if t is None.
            Returns:
                The perturbed image, and the corresponding sigma.
        '''
        if t is None:
            rnd_normal = torch.randn((x.shape[0], 1, 1, 1)).to(self.device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        else:
            times = reversed(self.sample_schedule(steps))
            sigma = times[t]
            if len(sigma.shape) == 1:
                sigma = sigma[:, None, None, None]

        noise = torch.randn_like(x)
        x_noised = x + noise * sigma
        return x_noised, sigma

    def forward(self, x, c, use_amp=False):
        ''' Training with weighted denoising loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
                c: The label for class-conditional generation.
            Returns:
                The weighted MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)

        x_noised, sigma = self.perturb(x, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss_4shape = weight * ((x - self.D_x(x_noised, sigma, (c, mask), use_amp)) ** 2)
        return loss_4shape.mean()

    def edm_sample(self, n_sample, size, steps=18, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False):
        ''' Sampling with EDM sampler. Actual NFE is `3 * steps - 1`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        print('In this function: EDM Sampling!')
        model_args = self.prepare_condition_(n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return unnormalize_to_zero_to_one(x_next)
    
    def edm_sample_single_class(self, n_sample, size, class_label=0, steps=18, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False, seed=None):
        ''' Sampling with EDM sampler. Actual NFE is `3 * steps - 1`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        print('In this function: EDM Sampling!')
        model_args = self.prepare_single_class_condition_(class_label, n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        ##############################################
        if seed is not None:
            torch.manual_seed(seed)

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        # print('min', x_next.min(), 'max', x_next.max())
        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            # print('min', x_next.min(), 'max', x_next.max())

        return unnormalize_to_zero_to_one(x_next)
    
    def edm_sample_single_class_save(self, n_sample, size, class_label=0, steps=18, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False, seed=None):
        ''' Sampling with EDM sampler. Actual NFE is `3 * steps - 1`.
            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                class_label: The class label for conditioning.
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                A list of sampled image tensors ranged in `[0, 1]` at each step.
        '''
        print('In this function: EDM Sampling!')
        model_args = self.prepare_single_class_condition_(class_label, n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))

        if seed is not None:
            torch.manual_seed(seed)

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        intermediate_results = [unnormalize_to_zero_to_one(x_next)]  # 初始状态也包括在内

        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)):
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            # Add intermediate result to the list.
            intermediate_results.append(unnormalize_to_zero_to_one(x_next))

        return intermediate_results

    # adversarial warm-up
    def edm_advprior_single_class(self, n_sample, size, classifier, class_label=0, steps=18, drop_rate=1e-2, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False, seed=None):

        model_args = self.prepare_single_class_condition_(class_label, n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        if seed is not None:
            torch.manual_seed(seed)
            
        from torchattacks import PGDL2
        from copy import deepcopy
        attack = PGDL2(model=classifier, eps=0.1, alpha=0.02, steps=10, random_start=True)
        
        def denoise(x_next):
            # print('min', x_next.min(), 'max', x_next.max())
            for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

                # Euler step.
                d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < steps - 1:
                    d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                # print('min', x_next.min(), 'max', x_next.max())
            return  x_next

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        seed = deepcopy(x_next)
        x_next = denoise(x_next)
        for i in range(8):
            min_val = x_next.min()
            max_val = x_next.max()
            x_next = (x_next - min_val) / (max_val - min_val)
            x_next_back = deepcopy(x_next)
            x_next = attack(x_next.to(torch.float32), torch.full((x_next.size(0),), class_label, dtype=torch.long)).to(torch.float64)
            perturbation = x_next - x_next_back
            min_val = seed.min()
            max_val = seed.max()
            seed = (seed - min_val) / (max_val - min_val)
            seed = seed + perturbation
            seed = seed * (max_val - min_val) + min_val
            x_next = denoise(seed)

        return unnormalize_to_zero_to_one(x_next)
    
    def edm_sample_hcseed_single_class(self, n_sample, size, classifier, class_label=0, steps=18, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False, seed=None, ball=0.1, step_size=0.001, n_rounds=1000):

        model_args = self.prepare_single_class_condition_(class_label, n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        def denoise(x_next):
            # print('min', x_next.min(), 'max', x_next.max())
            for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

                # Euler step.
                d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < steps - 1:
                    d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                # print('min', x_next.min(), 'max', x_next.max())
            return  x_next
        
        if seed is not None:
            torch.manual_seed(seed)

        from copy import deepcopy
        seed_vector = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]
        x_initial = deepcopy(seed_vector)
        x_initial = denoise(x_initial)
        # confidence = torch.softmax(classifier(unnormalize_to_zero_to_one(x_initial).to(torch.float32)), dim=-1)
        # print(confidence)
        confidence_initial = torch.softmax(classifier(unnormalize_to_zero_to_one(x_initial).to(torch.float32)), dim=-1)[:, class_label]
        
        def optimize_seed_vectorized(seed_vector, classifier, class_label, step_size, ball, n_rounds, device):
            total_noise = torch.zeros_like(seed_vector).to(device).to(torch.float64)  # 初始化总扰动噪声ε为0
            confidence_old = confidence_initial

            for round in range(n_rounds):
                print(round,':',confidence_old)
                noise = torch.randn_like(seed_vector).to(device).to(torch.float64) * step_size
                new_total_noise = torch.clamp(total_noise + noise, min=-ball, max=ball)
                # print(seed_vector[0][0])
                min_val = seed_vector.min()
                max_val = seed_vector.max()
                new_seeds = (seed_vector - min_val) / (max_val - min_val)
                new_seeds = new_seeds + new_total_noise
                new_seeds = new_seeds * (max_val - min_val) + min_val
                # print(new_seeds[0][0])
                x_next_new = denoise(new_seeds) 
                logits_new = classifier(unnormalize_to_zero_to_one(x_next_new).to(torch.float32))
                confidence_new = torch.softmax(logits_new, dim=-1)[:, class_label]
                
                update_mask = confidence_new < confidence_old
                total_noise = torch.where(update_mask.unsqueeze(-1).unsqueeze(2).unsqueeze(3), new_total_noise, total_noise)
                
                confidence_old = confidence_new
            
            optimized_seed = seed_vector# + total_noise
            return x_next_new, optimized_seed, total_noise
        
        x_next_new, _, _ = optimize_seed_vectorized(seed_vector=seed_vector, classifier=classifier, class_label=class_label, step_size=step_size, ball=ball, n_rounds=n_rounds, device=self.device)

        return unnormalize_to_zero_to_one(x_next_new)
        # return unnormalize_to_zero_to_one(x_initial)

    # adversarial guidance
    def edm_advdiff_single_class(self, n_sample, size, classifier, class_label=0, steps=18, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False):
        ''' Sampling with EDM sampler. Actual NFE is `3 * steps - 1`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        print('In this function: EDM Advdiff!')
        
        # prepare attack
        from torchattacks import PGDL2
        
        model_args = self.prepare_single_class_condition_(class_label, n_sample)
        S_min, S_max, S_noise = self.S_min, self.S_max, self.S_noise
        gamma_stochasticity = torch.tensor(np.sqrt(2) - 1) * eta # S_churn = (sqrt(2) - 1) * eta * steps

        times = self.sample_schedule(steps)
        time_pairs = list(zip(times[:-1], times[1:]))

        x_next = torch.randn(n_sample, *size).to(self.device).to(torch.float64) * times[0]

        attack = PGDL2(model=classifier, eps=0.02, alpha=0.005, steps=10, random_start=True)
        for i, (t_cur, t_next) in enumerate(tqdm(time_pairs, disable=notqdm)): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = gamma_stochasticity if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            d_cur = self.pred_eps_cfg(x_hat, t_hat, model_args, guide_w, use_amp)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                d_prime = self.pred_eps_(x_next, t_next, model_args, use_amp)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                
            # print('min', x_next.min(), 'max', x_next.max())

            if i < steps:
                min_val = x_next.min()
                max_val = x_next.max()
                x_next = (x_next - min_val) / (max_val - min_val)
                x_next = attack(x_next.to(torch.float32), torch.full((x_next.size(0),), class_label, dtype=torch.long)).to(torch.float64)
                x_next = x_next * (max_val - min_val) + min_val

                print('step',i,'attack!')
            else:
                print('step',i,'no attack!')

        return unnormalize_to_zero_to_one(x_next)

    def pred_eps_cfg(self, x, t, model_args, guide_w, use_amp, clip_x=True):
        x_double = x.repeat(2, 1, 1, 1)
        denoised = self.D_x(x_double, t, model_args, use_amp).to(torch.float64)
        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
        eps = (x_double - denoised) / t
        n_sample = eps.shape[0] // 2
        eps1 = eps[:n_sample]
        eps2 = eps[n_sample:]
        assert eps1.shape == eps2.shape
        eps = (1 + guide_w) * eps1 - guide_w * eps2
        return eps

    def pred_eps_(self, x, t, model_args, use_amp, clip_x=True):
        n_sample = x.shape[0]
        denoised = self.D_x(x, t, (model_args[0][:n_sample], model_args[1][:n_sample]), use_amp).to(torch.float64)
        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
        eps = (x - denoised) / t
        return eps

    def D_x(self, x_noised, sigma, model_args, use_amp):
        ''' Denoising with network preconditioning.

            Args:
                x_noised: The perturbed image tensor.
                sigma: The variance (noise level) tensor.
                model_args: class-conditional labels and drop masks.
            Returns:
                The estimated denoised image tensor.
        '''
        x_noised = x_noised.to(torch.float32)
        sigma = sigma.to(torch.float32)

        # Preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Denoising
        with autocast(enabled=use_amp):
            F_x = self.nn_model(c_in * x_noised, c_noise.flatten(), *model_args)
        return c_skip * x_noised + c_out * F_x

    def sample_schedule(self, steps):
        ''' Make the variance schedule for EDM sampling.

            Args:
                steps: The number of total timesteps. Typically 18, 50 or 100.
            Returns:
                times: A decreasing tensor list such that
                    `times[0] == sigma_max`,
                    `times[steps-1] == sigma_min`, and
                    `times[steps] == 0`.
        '''
        sigma_min, sigma_max, rho = self.sigma_min, self.sigma_max, self.rho
        times = torch.arange(steps, dtype=torch.float64, device=self.device)
        times = (sigma_max ** (1 / rho) + times / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        times = torch.cat([times, torch.zeros_like(times[:1])]) # t_N = 0
        return times

    def prepare_condition_(self, n_sample):
        n_classes = self.nn_model.num_classes
        assert n_sample % n_classes == 0
        c = torch.arange(n_classes).to(self.device)
        c = c.repeat(n_sample // n_classes)
        c = c.repeat(2)

        # 0 for conditional, 1 for unconditional
        mask = torch.zeros_like(c).to(self.device)
        mask[n_sample:] = 1.
        return c, mask
    
    def prepare_single_class_condition_(self, class_label, n_sample):
        assert 0 <= class_label < self.nn_model.num_classes, "Class label out of range"

        c = torch.full((n_sample,), class_label, dtype=torch.long).to(self.device)
        c = c.repeat(2)

        mask = torch.zeros_like(c).to(self.device)
        mask[n_sample:] = 1.
        
        return c, mask
