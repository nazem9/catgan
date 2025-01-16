import torch
import torch.nn as nn
import torchsummary
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """Pixel normalization layer."""
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Calculate the squared sum of features along the channel dimension
        squared_sum = torch.sum(x**2, dim=1, keepdim=True)
        # Normalize each pixel
        normalized = x / torch.sqrt(squared_sum + self.epsilon)
        return normalized

class ResidualBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super(ResidualBlockGenerator, self).__init__()
        layers = []

        # First layer (may change the number of channels)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(PixelNorm())
        layers.append(nn.LeakyReLU(0.2))

        # Additional layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(PixelNorm())
            layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

        # Adjust skip connection if the number of input and output channels differ
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                PixelNorm()
            )
        else:
            self.skip_connection = None

        self.activation = nn.LeakyReLU(0.2)  # Use LeakyReLU consistently

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        out += identity
        return self.activation(out)
    
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.init_size = 2  # Starting with 2x2 feature maps
        self.latent_dim = latent_dim

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 1024 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2)
        )

        self.conv_blocks = nn.Sequential(
            # After reshaping, feature map size: (1024, 2, 2)

            # Upsample to (512, 4, 4)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Upsample to (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Upsample to (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Upsample to (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Upsample to (32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 2048, self.init_size, self.init_size)  # Reshape to (batch_size, 1024, 2, 2)
        img = self.conv_blocks(out)
        return img  # Output shape: [batch_size, 3, 64, 64]

# Minimal Convolutional Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_map_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

			# 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
			# 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
			
			nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

			nn.Flatten(),
            # Output layer
            nn.Linear(feature_map_size * 4 * 32, 1),
        )

    def forward(self, img):
        out = self.disc(img)
        return out.view(-1, 1)  # Output shape: [batch_size, 1]

# Example usage
if __name__ == '__main__':
    # Hyperparameters
    latent_dim = 100
    batch_size = 8
    img_channels = 3
    img_size = 64  # Image size: 64x64

    # Create noise vector
    noise = torch.randn(batch_size, latent_dim,device='cuda')

    # Instantiate models
    gen = Generator(latent_dim=latent_dim).to('cuda')
    disc = Discriminator(img_channels=img_channels).to('cuda')

    # Generate fake images
    fake_images = gen(noise)
    print("Fake image size:", fake_images.size())

    # Pass fake images through discriminator
    disc_output = disc(fake_images)

    # Print shapes to verify
    print(f"Fake images shape: {fake_images.shape}")          # Expected: [batch_size, 3, 64, 64]
    print(f"Discriminator output shape: {disc_output.shape}")  # Expected: [batch_size, 1]
    
    print(torchsummary.summary(gen,(100,)))
    print(torchsummary.summary(disc,(3,64,64)))