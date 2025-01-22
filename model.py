import torch
import torch.nn as nn
import torchsummary

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Normalize the feature vector in each pixel to unit length
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = 11  # Initial size to eventually reach 178x178
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build the generator network with upsampling layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            
            # Upsample from 11x11 -> 22x22
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample from 22x22 -> 44x44
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample from 44x44 -> 88x88
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample from 88x88 -> 176x176
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Conv layer to reach 178x178
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # Output size: 176x176
            nn.Upsample(size=(178, 178)),  # Upsample to 178x178
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 512, self.init_size, self.init_size)  # Reshape to (batch_size, 512, 11, 11)
        img = self.conv_blocks(out)
        return img  # Output size: (batch_size, 3, 178, 178)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Input size: (3, 178, 178)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),    # Output: (64, 89, 89)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 45, 45)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Output: (256, 23, 23)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Output: (512, 12, 12)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # Output: (1, 1, 1)
            nn.Flatten(),
            nn.Linear(30976, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)  # Output size: (batch_size, 1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 100
    batch_size = 8
    img_channels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Test forward pass
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_imgs = generator(z)
    disc_output = discriminator(fake_imgs)

    # Print shapes
    print(f"Generator output shape: {fake_imgs.shape}")  # Expected: [8, 3, 178, 178]
    print(f"Discriminator output shape: {disc_output.shape}")  # Expected: [8, 1]

    # Print model summaries
    print("\nGenerator Summary:")
    torchsummary.summary(generator, (latent_dim,), device=str(device))
    print("\nDiscriminator Summary:")
    torchsummary.summary(discriminator, (3, 178, 178), device=str(device))