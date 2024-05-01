import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from robustbench import load_model
from robustbench.data import load_cifar10, load_cifar100
from model import create_model, Normalize, PreActResNet18
from tqdm import tqdm
import argparse
from spgd import SparsePGD
import os
from rs_attacks import RSAttack


# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def load_mnist(data_dir, n_examples):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
    x_test = test_dataset.data[:n_examples].unsqueeze(1).float() / 255.
    y_test = test_dataset.targets[:n_examples]

    return x_test, y_test


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--ckpt', type=str,
                    help='For robustbench, it is the name. For others, it is the path of checkpoint')
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.25, help='real alpha = alpha * eps')
parser.add_argument('--beta', type=float, default=0.25, help='real beta = beta * sqrt(h*w)')
parser.add_argument('--n_iters', type=int, default=300)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--n_examples', type=int, default=10000)
parser.add_argument('--model', type=str, choices=['standard', 'linf', 'l0'], default='standard')
parser.add_argument('--unprojected_gradient', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

save_path = os.path.join('exp', f'{args.exp}_{args.model}_k{args.k}_{args.n_examples}examples_{args.n_iters}iters')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# loading data
x_test, y_test = load_mnist(n_examples=args.n_examples, data_dir=args.data_dir)
dataset = TensorDataset(x_test, y_test)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)
num_classes = 10


# model
net = create_model(name='mnist', num_classes=num_classes, channel=1, norm='batchnorm')
net.load_state_dict(torch.load(args.ckpt)['state_dict'])

net = net.cuda()
net.eval()

clean_acc = 0.
robust_acc = 0.
robust_acc_ut = 0.
attacker = SparsePGD(net, epsilon=args.eps, k=args.k, t=args.n_iters, unprojected_gradient=args.unprojected_gradient,
                     classes=num_classes)
x_for_white2, y_for_white2 = None, None
print('White-box attack 1')
for x, y in tqdm(loader):
    x = x.cuda()
    y = y.cuda()

    # # Untargeted + 9 x targeted attack
    # c_acc, r_acc, r_acc_ut, ind_fail = attacker.auto_attack(x, y)
    # clean_acc += c_acc
    # robust_acc += r_acc
    # robust_acc_ut += r_acc_ut

    # Only untargeted attack
    x_adv, mask = attacker.perturb(x, y)
    fool_label = torch.argmax(net(x_adv), dim=1)
    clean_label = torch.argmax(net(x), dim=1)

    clean_acc += (clean_label == y).float().sum().item()
    robust_acc += (fool_label == y).float().sum().item()
    ind_fail = (fool_label == y).nonzero().squeeze()

    ind_success = ((fool_label != y) * (clean_label == y)).nonzero().squeeze()
    img_sucess = x_adv[ind_success]
    img_sucess = img_sucess[0]
    img_sucess = img_sucess.permute(1, 2, 0)
    img_sucess = img_sucess.cpu().numpy()
    original = x[ind_success]
    original = original[0]
    original = original.permute(1, 2, 0)
    original = original.cpu().numpy()
    import matplotlib.pyplot as plt

    plt.imshow(original)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('clean_class{}_k{}_{}.png'.format(clean_label[ind_success][0].item(), args.k, args.model))
    plt.imshow(img_sucess)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('adv_class{}_k{}_{}.png'.format(fool_label[ind_success][0].item(), args.k, args.model))


#     # Save index for black-box attack 2
#     if ind_fail.numel() > 0:
#         x_fail, y_fail = x[ind_fail], y[ind_fail]
#         if x_fail.dim() == 3:
#             x_fail = x_fail.unsqueeze(0)
#             y_fail = y_fail.unsqueeze(0)
#         x_for_white2 = x_fail if x_for_white2 is None else torch.cat((x_for_white2, x_fail), dim=0)
#         y_for_white2 = y_fail if y_for_white2 is None else torch.cat((y_for_white2, y_fail), dim=0)
#         torch.save(ind_fail.cpu(), os.path.join(save_path, 'white1_ind_fail.pth'))
#
# clean_acc = round(clean_acc / len(dataset), 4)
# robust_acc = round(robust_acc / len(dataset), 4)
# robust_acc_ut = round(robust_acc_ut / len(dataset), 4)
# print('Clean Acc:', clean_acc)
# print('Robust Acc after untargeted attack:', robust_acc_ut)
# print('Robust Acc after white-box attack 1:', robust_acc)
#
# if x_for_white2 is not None:
#     x_for_white2 = x_for_white2.cpu()
#     y_for_white2 = y_for_white2.cpu()
#     dataset_for_white2 = TensorDataset(x_for_white2, y_for_white2)
#     loader_for_white2 = DataLoader(dataset_for_white2, batch_size=min(len(dataset_for_white2), 500), shuffle=False,
#                                    num_workers=2)
#     print('# White-box 1 attack failed samples:', len(dataset_for_white2))
#     print('White-box attack 2')
#     robust_acc_for_white2 = 0.
#     attacker = SparsePGD(net, epsilon=args.eps, k=args.k, t=args.n_iters, unprojected_gradient=not args.unprojected_gradient,
#                          classes=num_classes)
#     x_for_black, y_for_black = None, None
#
#     for x, y in tqdm(loader_for_white2):
#         x = x.cuda()
#         y = y.cuda()
#
#         # Untargeted + 9 x targeted attack
#         c_acc, r_acc, _, ind_fail = attacker.auto_attack(x, y)
#         robust_acc_for_white2 += r_acc
#
#         # Only untargeted attack
#         # x_adv, mask = attacker.perturb(x, y)
#         # fool_label = torch.argmax(net(x_adv), dim=1)
#         # clean_label = torch.argmax(net(x), dim=1)
#
#         # robust_acc_for_white2 += (fool_label == y).float().sum().item()
#         # ind_fail = (fool_label == y).nonzero().squeeze()
#
#         if ind_fail.numel() > 0:
#             x_fail, y_fail = x[ind_fail], y[ind_fail]
#             if x_fail.dim() == 3:
#                 x_fail = x_fail.unsqueeze(0)
#                 y_fail = y_fail.unsqueeze(0)
#             x_for_black = x_fail if x_for_black is None else torch.cat((x_for_black, x_fail), dim=0)
#             y_for_black = y_fail if y_for_black is None else torch.cat((y_for_black, y_fail), dim=0)
#             torch.save(ind_fail.cpu(), os.path.join(save_path, 'white2_ind_fail.pth'))
#
#     robust_acc = round(robust_acc * robust_acc_for_white2 / len(dataset_for_white2), 4)
#     print('Robust Acc after white-box attack 2:', robust_acc)
#
#     if x_for_black is not None:
#
#         x_for_black = x_for_black.cpu()
#         y_for_black = y_for_black.cpu()
#         dataset_for_black = TensorDataset(x_for_black, y_for_black)
#         loader_for_black = DataLoader(dataset_for_black, batch_size=min(len(dataset_for_black), 500), shuffle=False,
#                                       num_workers=2)
#         print('# White-box 2 attack failed samples:', len(dataset_for_black))
#         print('Black-box attack')
#         attack = RSAttack(net, norm='L0', eps=args.k, verbose=False, n_queries=3000, p_init=0.3, targeted=False)
#
#         robust_acc_for_black = 0.
#         with torch.no_grad():
#             for x, y in tqdm(loader_for_black):
#                 x = x.cuda()
#                 y = y.cuda()
#                 qr_curr, adv = attack.perturb(x, y)
#
#                 with torch.no_grad():
#                     perturb = adv - x
#                     assert torch.norm(perturb.sum(1), p=0, dim=(1, 2)).max().item() <= args.k, 'projection error'
#                     assert torch.max(adv).item() <= 1.0 and torch.min(
#                         adv).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(
#                         torch.min(adv).item(),
#                         torch.max(adv).item())
#
#
#                 output = net(adv.cuda())
#                 robust_acc_for_black += (output.max(1)[1] == y).float().sum().item()
#                 ind_succ = (output.max(1)[1] != y).nonzero().squeeze()
#
#                 ind_fail = (output.max(1)[1] == y).nonzero().squeeze()
#                 if ind_fail.numel() > 0:
#                     x_fail, y_fail = x[ind_fail], y[ind_fail]
#                     if x_fail.dim() == 3:
#                         x_fail = x_fail.unsqueeze(0)
#                         y_fail = y_fail.unsqueeze(0)
#                     torch.save(ind_fail.cpu(), os.path.join(save_path, 'black_ind_fail.pth'))
#
#         robust_acc = round(robust_acc * robust_acc_for_black / len(dataset_for_black), 4)
#         print('Robust Acc after black-box attack:', robust_acc)

