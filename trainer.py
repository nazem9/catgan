import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms import  v2
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
import kagglehub

path = kagglehub.dataset_download("mahmudulhaqueshawon/cat-image")
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

BATCH_SIZE = 128

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKER = int(os.cpu_count() / 2)

from PIL import Image
tsf = v2.Compose([
    transforms.PILToTensor(),
    transforms.CenterCrop(128),
    transforms.Resize((128,128)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std = (0.5, 0.5, 0.5))
                         ])


class HumanFacesDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),                 # Slightly larger for random crop
            # transforms.RandomCrop((64, 64)),             # Randomly crop to desired size
            # transforms.RandomHorizontalFlip(),           # Randomly flip images horizontally
            transforms.ToTensor(),                       # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5),        # Normalize images to [-1, 1]
                                 (0.5, 0.5, 0.5))
        ])

    def setup(self, stage: str = None):
        """
        Setup datasets for different stages: 'fit', 'test', or 'predict'.
        Split the dataset into training, validation, and test sets.
        """
        self.full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        # test_size = int(len(full_dataset) * 0.2)
        # val_size = int(len(full_dataset) * 0.1)
        # train_size = len(full_dataset) - test_size - val_size
        
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(
        #     full_dataset, [train_size, val_size, test_size]
        # )

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True) #, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



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
data_dir = "/kaggle/input/cat-image" if os.path.isdir(r"/kaggle/input/cat-image" ) else path
data_module = HumanFacesDataModule(data_dir=data_dir, batch_size=32, num_workers=15)

                    
model = GAN(latent=64,    g_lr=1e-4,
    d_lr=1e-4,
    channels=1,
    scheduler_type='cosine',
    T_max=200,  # If None, will be set to max_epochs
    T_0=10,
    T_mult=2,
    eta_min=1e-6)

trainer = pl.Trainer(max_epochs = 200 , callbacks=get_callbacks(dirpath="./checkponts1"), logger=tensorboard_logger,
                     accelerator='gpu',
                        devices=1,)
trainer.fit(model,data_module)