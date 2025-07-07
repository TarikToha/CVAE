import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA, VisionDataset
from torchvision.datasets.folder import default_loader


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class RadarDataset(VisionDataset):

    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

        self.data = glob.glob(f'{root}/*.jpg')

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.open(img)
        img = self.transform(img)
        return img, 0.0

    def __len__(self):
        return len(self.data)


class VisionRadarDataset(VisionDataset):

    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

        data_A = glob.glob(f'{root}/A/*.jpg')
        self.data = []
        for file_path_A in data_A:
            file_name = os.path.basename(file_path_A)
            file_path_B = f'{root}/B/{file_name}'
            self.data.append((file_path_A, file_path_B))

    def __getitem__(self, index):
        img_A, img_B = self.data[index]
        img_A, img_B = Image.open(img_A), Image.open(img_B)
        img_A, img_B = self.transform(img_A), self.transform(img_B)

        img = torch.cat([img_A, img_B], dim=0)
        return img, 0.0

    def __len__(self):
        return len(self.data)


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            dataset_name: str,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        train_transforms = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # =========================  Radar Dataset  =========================
        if self.dataset_name == "RadarDataset":
            self.train_dataset = RadarDataset(
                root=f'{self.data_dir}train',
                transform=train_transforms,
            )

            self.val_dataset = RadarDataset(
                root=f'{self.data_dir}test',
                transform=val_transforms,
            )

        # =========================  VisionRadar Dataset  =========================
        elif self.dataset_name == "VisionRadarDataset":
            self.train_dataset = VisionRadarDataset(
                root=f'{self.data_dir}train',
                transform=train_transforms,
            )

            self.val_dataset = VisionRadarDataset(
                root=f'{self.data_dir}test',
                transform=val_transforms,
            )
        else:
            print("Dataset not recognized")
            exit()

        #       =========================  CelebA Dataset  =========================
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                        transforms.CenterCrop(148),
        #                                        transforms.Resize(self.patch_size),
        #                                        transforms.ToTensor(), ])
        #
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                      transforms.CenterCrop(148),
        #                                      transforms.Resize(self.patch_size),
        #                                      transforms.ToTensor(), ])
        #
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        #
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
