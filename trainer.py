import os
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import kagglehub
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
path = os.path.join(path, "img_align_celeba","img_align_celeba")
print("Path to dataset files:", path)
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl

from pytorch_lightning import LightningDataModule

from lightning_module import GAN

from pytorch_lightning.loggers import WandbLogger
import datetime
date  = datetime.datetime.strftime(datetime.datetime.now(),"%H%M%S")
# Initialize TensorBoard logger
tensorboard_logger = WandbLogger(
    # save_dir='tb_logs',      # Directory to save TensorBoard logs
    name=f'GAN_Experiments {date}',  # Name of the experiment
    version=f"version_{date}"     # (Optional) Version of the experiment
)

BATCH_SIZE = 32

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKER = int(os.cpu_count() / 2)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset to load images directly from a folder without class subfolders.
        Args:
            root_dir (str): Path to the folder containing images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)

        return image, 0  # Return a dummy label (0), as we don't use labels in GANs


class HumanFacesDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((178, 178)),                 # Resize images to 64x64
            transforms.ToTensor(),                       # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5),        # Normalize images to [-1, 1]
                                 (0.5, 0.5, 0.5))
        ])

    def setup(self, stage: str = None):
        """
        Setup datasets for different stages: 'fit', 'test', or 'predict'.
        Load images directly from the root directory without class subfolders.
        """
        self.full_dataset = ImageDataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size) #, shuffle=True, num_workers=self.num_workers)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def get_callbacks(dirpath='checkpoints'):
    callbacks = [
        # Model checkpoint callback
        ModelCheckpoint(
            dirpath=dirpath,
            filename='gan-{epoch:02d}-{g_loss:.2f}',
            monitor='g_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
        
        # Early stopping
        EarlyStopping(
            monitor='g_loss',
            patience=200,
            mode='min',
            verbose=True
        )
    ]
    return callbacks

# Example usage:
data_dir = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba" if os.path.isdir(r"/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba" ) else path
data_module = HumanFacesDataModule(data_dir=data_dir, batch_size=BATCH_SIZE, num_workers=12)

                    
model = GAN(g_lr=1e-4, d_lr=5e-4, disc_steps=5,latent=64,lambda_gp=20)

trainer = pl.Trainer(max_epochs = 200 , callbacks=get_callbacks(dirpath="./checkponts1"), logger=tensorboard_logger,
                     accelerator='gpu',
                        devices=1,)

trainer.fit(model,data_module)