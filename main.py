import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataset import SkinDataset
from model import Generator, Discriminator
from solver import Solver
from logger import Logger  

def main(config):
    # Create directories if they don't exist
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Initialize logger
    if config.use_tensorboard:
        logger = Logger(log_dir=config.log_dir)
    else:
        logger = None

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming normalization based on dataset stats
    ])

    # Create data loaders
    train_dataset = SkinDataset(root_dir=os.path.join(config.dataset_root, 'train_set'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    test_dataset = SkinDataset(root_dir=os.path.join(config.dataset_root, 'test_set'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize models
    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num)
    D = Discriminator(image_size=config.image_size, conv_dim=config.d_conv_dim, c_dim=config.c_dim, repeat_num=config.d_repeat_num)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    # Optimizers
    g_optimizer = optim.Adam(G.parameters(), config.g_lr, [config.beta1, config.beta2])
    d_optimizer = optim.Adam(D.parameters(), config.d_lr, [config.beta1, config.beta2])

    # Solver - Corrected instantiation with instances G and D
    solver = Solver(train_loader, test_loader, G, D, g_optimizer, d_optimizer, device, config, logger)


    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    # Close logger if used
    if logger is not None:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Training configuration
    parser.add_argument('--dataset_root', type=str, default='dataset', help='root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Miscellaneous
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='whether to use TensorBoard for logging')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory for TensorBoard logs')
    parser.add_argument('--model_save_dir', type=str, default='models', help='directory for saving models')
    parser.add_argument('--sample_dir', type=str, default='samples', help='directory for saving samples')
    parser.add_argument('--result_dir', type=str, default='results', help='directory for saving results')

    config = parser.parse_args()
    print(config)
    main(config)
