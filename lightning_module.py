import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from model import Generator, Discriminator


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GAN(pl.LightningModule):
    def __init__(
        self, 
        latent=100, 
        g_lr=1e-4, 
        d_lr=2e-4, 
        channels=3, 
        disc_steps=5, 
        lambda_gp=10, 
        weight_clip=0.01,
        t_0=10,  # Number of iterations for the first restart
        t_mult=2,  # Factor to increase t_0 after each restart
        eta_min=1e-6  # Minimum learning rate
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.validation_z = torch.randn(6, self.hparams.latent)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(d_interpolates.size(), device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        opt_d, opt_g = self.optimizers()

        device = real_imgs.device

        # Train discriminator
        for _ in range(self.hparams.disc_steps):
            z = torch.randn(batch_size, self.hparams.latent, device=device)
            fake_imgs = self(z)

            fake_validity = self.discriminator(fake_imgs)
            real_validity = self.discriminator(real_imgs)

            gp = self.compute_gradient_penalty(real_imgs, fake_imgs)
            d_loss = real_validity.mean() - fake_validity.mean() + self.hparams.lambda_gp * gp

            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

        # Train generator
        z = torch.randn(batch_size, self.hparams.latent, device=device)
        fake_imgs = self(z)
        fake_validity = self.discriminator(fake_imgs)

        g_loss = -fake_validity.mean()

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()


        # Logging
        self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)
        self.log('g_loss', g_loss, prog_bar=True, on_epoch=True)
        self.log('lr_d', opt_d.param_groups[0]['lr'], prog_bar=True)
        self.log('lr_g', opt_g.param_groups[0]['lr'], prog_bar=True)

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.hparams.d_lr, 
            betas=(0.0, 0.9)
        )
        opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.hparams.g_lr, 
            betas=(0.0, 0.9)
        )
        
        # Cosine Annealing schedulers
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_d,
            T_0=self.hparams.t_0,
            T_mult=self.hparams.t_mult,
            eta_min=self.hparams.eta_min
        )
        
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_g,
            T_0=self.hparams.t_0,
            T_mult=self.hparams.t_mult,
            eta_min=self.hparams.eta_min
        )

        return [opt_d, opt_g], [scheduler_d, scheduler_g]

    def plot_images(self):
        z = self.validation_z.to(self.device)
        sample_images = self(z).cpu()

        sample_images = (sample_images + 1) / 2
        os.makedirs("./generated_images", exist_ok=True)
        save_image(sample_images, f'generated_images/epoch_{self.current_epoch}.png', nrow=3, normalize=True)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
		# Note: pl_module parameter was removed since we're inside the module itself
        sch_d, sch_g = self.lr_schedulers()        
        if self.global_step % 50 == 0:
			# Step the schedulers
            sch_d.step()
            sch_g.step()

    def on_train_epoch_end(self):
        self.plot_images()
        print(f"Epoch {self.current_epoch} completed.")