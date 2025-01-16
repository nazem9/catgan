from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import torch
# from model import Discriminator, Generator
import torch.nn.functional as F

from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import torch
from model import Discriminator, Generator
import torch.nn.functional as F
import torch.nn as nn

import os
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GAN(pl.LightningModule):
    def __init__(self, latent=100, 
                 g_lr=1e-4, 
                 d_lr=2e-4, 
                 channels=3,
                 scheduler_type='cosine',  # 'cosine' or 'cosine_warm'
                 T_max=100,  # For regular cosine annealing
                 T_0=10,     # For warm restarts
                 T_mult=2,   # For warm restarts
                 eta_min=1e-8,
                  lr_lambda=None):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        # Random noise for generating images at the end of each epoch
        self.validation_z = 0.5 + 0.5 * torch.randn(6, self.hparams.latent)

        # Enable manual optimization
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Get optimizers
        opt_d, opt_g = self.optimizers()

        # Sample noise
        z = 0.5 + 0.5 * torch.randn(batch_size, self.hparams.latent, device=device)
        fake_imgs = self(z)

        # Train Discriminator
        # -------------------

        # Forward passes
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs.detach())

        # Real labels are ones, fake labels are zeros
        y_real = torch.ones(batch_size, 1, device=device)
        y_fake = torch.zeros(batch_size, 1, device=device)

        # Compute discriminator loss
        real_loss = self.adversarial_loss(real_validity, y_real)
        fake_loss = self.adversarial_loss(fake_validity, y_fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backward and optimization step for discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Train Generator
        # ----------------

        # Generate fake images again (since we detached earlier)
        fake_imgs = self(z)
        fake_validity = self.discriminator(fake_imgs)

        # Generator wants discriminator to think these are real
        g_loss = self.adversarial_loss(fake_validity, y_real)

        # Backward and optimization step for generator
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Logging
        self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)
        self.log('g_loss', g_loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # Optimizers
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.hparams.d_lr, 
            betas=(0.5, 0.999),
            weight_decay=1e-3 # Optional: weight decay
        )
        opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.hparams.g_lr, 
            betas=(0.9, 0.999),
            weight_decay=1e-3 # Optional: weight decay
        )
        
        # Learning rate schedulers
        # Learning rate schedulers
        scheduler_d = {
            'scheduler': torch.optim.lr_scheduler.StepLR(opt_d, gamma=0.99, step_size=1),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'scheduler_d'
        }
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.StepLR(opt_g, gamma=0.99, step_size=1),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'scheduler_g'
        }
        
        return [opt_d, opt_g], [scheduler_d, scheduler_g]

    def plot_images(self):
        z = self.validation_z.to(self.device)
        sample_images = self(z).cpu()

        # Rescale images from [-1, 1] to [0, 1]
        sample_images = (sample_images + 1) / 2
        os.makedirs("./generated_images2", exist_ok=True)
        # Save images
        save_image(sample_images, 
                   f'generated_images2/epoch_{self.current_epoch}.png', 
                   nrow=3, 
                   normalize=True)
        
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     pass

    def on_train_epoch_end(self):
        self.plot_images()
        print(f"Epoch {self.current_epoch} completed.")
        # Manually step schedulers every odd epoch
        if self.current_epoch % 20 == 0:
            schedulers = self.lr_schedulers()
            for scheduler in schedulers:
                scheduler.step()
