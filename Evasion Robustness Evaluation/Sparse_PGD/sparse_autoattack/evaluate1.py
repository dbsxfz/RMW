import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from robustbench import load_model
from robustbench.data import load_cifar10
from model import create_model, Normalize, PreActResNet18
from tqdm import tqdm
import argparse
from spgd import SparsePGD
import os
from rs_attacks import RSAttack
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--ckpt', type=str, help='For robustbench, it is the name. For others, it is the path of checkpoint')
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.25, help='real alpha = alpha * eps')
parser.add_argument('--beta', type=float, default=0.25, help='real beta = beta * sqrt(h*w)')
parser.add_argument('--n_iters', type=int, default=300)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--n_examples', type=int, default=10000)
parser.add_argument('--model', type=str, choices=['standard', 'l1', 'l2', 'linf', 'l0'], default='l1')
parser.add_argument('--original_for_back', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

save_path = os.path.join('exp', f'{args.exp}_{args.model}_k{args.k}_{args.n_examples}examples_{args.n_iters}iters')
if not os.path.exists(save_path):
    os.makedirs(save_path)

x_test, y_test = load_cifar10(n_examples=args.n_examples, data_dir=args.data_dir)
dataset = TensorDataset(x_test, y_test)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0)


# standard model
if args.model == 'standard':
    net = create_model(name='resnet18', num_classes=10, channel=3, norm='batchnorm')
    net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    net = nn.Sequential(
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        net
    )
# robust model
elif args.model == 'l2':
    net = load_model(args.ckpt, dataset='cifar10', threat_model='L2')
elif args.model == 'linf':
    net = load_model(args.ckpt, dataset='cifar10', threat_model='Linf')
elif args.model == 'l1':
    net = PreActResNet18(n_cls=10, activation='softplus1')
    ckpt2load = torch.load(args.ckpt)
    ckpt = {}
    for k, v in ckpt2load.items():
        ckpt[k[2:]] = v
    net.load_state_dict(ckpt)
elif args.model == 'l0':
    net = PreActResNet18(n_cls=10, activation='softplus1')
    net = nn.Sequential(
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        net
    )
    net.load_state_dict(
        torch.load(args.ckpt)['state_dict'])
else:
    raise NotImplementedError


net = net.cuda()
net.eval()

clean_acc = 0.
robust_acc = 0.
robust_acc_ut = 0.
attacker = SparsePGD(net, epsilon=args.eps, k=args.k, t=args.n_iters, unprojected_gradient=args.unprojected_gradient)
x_for_white2, y_for_white2 = None, None
print('White-box attack 1')
for x, y in tqdm(loader):
    x = x.cuda()
    y = y.cuda()

    # Untargeted + 9 x targeted attack
    c_acc, r_acc, robust_acc_ut, ind_fail = attacker.ensemble_attack(x, y)
    clean_acc += c_acc
    robust_acc += r_acc
    robust_acc_ut += robust_acc_ut
    
    # Only untargeted attack
    # x_adv, mask = attacker.perturb(x, y)
    # fool_label = torch.argmax(net(x_adv), dim=1)
    # clean_label = torch.argmax(net(x), dim=1)
    #
    # clean_acc += (clean_label == y).float().sum().item()
    # robust_acc += (fool_label == y).float().sum().item()

#print('Black-box attack')
#attack = RSAttack(net, norm='L0', eps=args.k, verbose=False, n_queries=args.n_iters, p_init=0.3, targeted=False)
#with torch.no_grad():
#    for x, y in tqdm(loader):
#        x = x.cuda()
#        y = y.cuda()
#        qr_curr, adv = attack.perturb(x, y)
#        output = net(adv.cuda())
#        robust_acc += (output.max(1)[1] == y).float().sum().item()

clean_acc = round(clean_acc / len(dataset), 4)
robust_acc = round(robust_acc / len(dataset), 4)
robust_acc_ut = round(robust_acc_ut / len(dataset), 4)
print('Clean Acc:', clean_acc)
print('Robust Acc UT:', robust_acc_ut)
print('Robust Acc:', robust_acc)

