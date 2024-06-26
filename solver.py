import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from datetime import datetime
from model import Generator, Discriminator

class Solver(object):
    def __init__(self, train_loader, test_loader, G, D, g_optimizer, d_optimizer, device, config, logger=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.G = G.to(self.device)  # Use the passed Generator instance
        self.D = D.to(self.device)  # Use the passed Discriminator instance
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.use_tensorboard = config.use_tensorboard
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Model configurations
        self.lambda_cls = getattr(config, 'lambda_cls', 1.0)
        self.lambda_rec = getattr(config, 'lambda_rec', 10.0)
        self.lambda_gp = getattr(config, 'lambda_gp', 10.0)
        self.resume_iters = config.resume_iters if config.resume_iters is not None else 0

        # Optimizers
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        # Logging configurations
        if self.use_tensorboard and logger is not None:
            from logger import Logger
            self.logger = Logger(self.log_dir)
        else:
            self.logger = None

    def build_model(self):
        """Build generators and discriminators."""
        # Typically, you don't need to redefine the models here if they are passed during initialization.
        pass

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x):
        """Convert tensor to variable."""
        return x.to(self.device)

    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def compute_gradient_penalty(self, D, real, fake):
        """Compute gradient penalty for WGAN-GP."""
        alpha = torch.rand(real.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates, _ = D(interpolates)
        fake = torch.ones(d_interpolates.size()).to(self.device)

        gradients = torch.autograd.grad(outputs=d_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=fake,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def train(self):
        """Train the GAN."""

        # Data iterator
        data_iter = iter(self.train_loader)
        iter_per_epoch = len(self.train_loader)

        # Fixed input for debugging
        fixed_x, _ = next(data_iter)
        fixed_x = self.to_var(fixed_x)

        # Start training
        print('Start training...')
        start_time = datetime.now()

        labels_real = None  # Initialize labels_real outside the try-except block

        for step in range(self.resume_iters, self.num_iters):
            # Fetch the next batch of images
            try:
                x_real, labels_real = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x_real, labels_real = next(data_iter)

            # Convert tensor to variable
            x_real = self.to_var(x_real)
            labels_real = self.to_var(labels_real) if labels_real is not None else None  # Assuming labels are tensors

            # Ensure labels_real has the correct shape
            if labels_real is not None and len(labels_real.shape) == 1:
                labels_real = labels_real.unsqueeze(1)

            # Debug print statement to verify dimensions
            print(f"x_real shape: {x_real.shape}, labels_real shape: {labels_real.shape}")

            # Discriminator training
            x_fake = self.G(x_real, labels_real)    # Pass labels_real to the generator
            d_real, _ = self.D(x_real)
            d_fake, _ = self.D(x_fake.detach())

            # Compute loss for WGAN-GP
            d_loss_real = -torch.mean(d_real)
            d_loss_fake = torch.mean(d_fake)
            gradient_penalty = self.compute_gradient_penalty(self.D, x_real.data, x_fake.data)
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gradient_penalty

            # Backprop + Optimize
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Generator training
            if (step + 1) % self.n_critic == 0:
                x_fake = self.G(x_real, labels_real)
                d_fake, c_fake = self.D(x_fake)

                # Compute loss for GAN loss
                g_loss_fake = -torch.mean(d_fake)

                # Backprop + Optimize
                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()

                # Print out log info
                if (step + 1) % 100 == 0:
                    elapsed = datetime.now() - start_time
                    elapsed = str(elapsed).split('.')[0]

                    print(f"Step [{step+1}/{self.num_iters}], "
                          f"d_loss: {d_loss.item():.4f}, g_loss_fake: {g_loss_fake.item():.4f}, "
                          f"time: {elapsed}")

                # Logging
                if self.logger is not None:
                    self.logger.scalar_summary('d_loss', d_loss.item(), step + 1)
                    self.logger.scalar_summary('g_loss_fake', g_loss_fake.item(), step + 1)

                # Save generated images
                if (step + 1) % 1000 == 0:
                    fake_images = self.denorm(x_fake.data)
                    save_image(fake_images, os.path.join(self.sample_dir, f'fake_images-{step+1}.png'), nrow=4, padding=1)
