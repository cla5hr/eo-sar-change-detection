import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split, image_size=256):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], additional_targets={"sar": "image"})
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ], additional_targets={"sar": "image"})


def remap_mask(mask):
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[mask == 2] = 1
    new_mask[mask == 3] = 1
    return new_mask


def is_valid_patch(eo, sar, threshold=0.3):
    # Skip patches where more than 30% pixels are black
    eo_black = (eo.sum(axis=-1) == 0).mean()
    sar_black = (sar == 0).mean()
    return eo_black < threshold and sar_black < threshold


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=256, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform if transform else get_transforms(split, image_size)

        self.pre_dir  = os.path.join(root_dir, split, "pre-event")
        self.post_dir = os.path.join(root_dir, split, "post-event")
        self.mask_dir = os.path.join(root_dir, split, "target")

        all_files = sorted(os.listdir(self.pre_dir))

        self.files = all_files

        print(f"{split}: using all {len(all_files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        eo = np.array(
            Image.open(
                os.path.join(
                    self.pre_dir,
                    filename
                )
            )
        ).astype(np.uint8)

        sar = np.array(
            Image.open(
                os.path.join(
                    self.post_dir,
                    filename
                )
            )
        ).astype(np.uint8)

        mask = np.array(
            Image.open(
                os.path.join(
                    self.mask_dir,
                    filename
                )
            )
        ).astype(np.uint8)

        mask = remap_mask(mask)

        # Replace black corners with small noise
        eo_black = (eo.sum(axis=-1) == 0)

        sar_black = (sar == 0)

        if eo_black.any():

            noise = np.random.randint(
                1,
                15,
                eo.shape,
                dtype=np.uint8
            )

            eo[eo_black] = noise[eo_black]

        if sar_black.any():

            noise = np.random.randint(
                1,
                10,
                sar.shape,
                dtype=np.uint8
            )

            sar[sar_black] = noise[sar_black]

        augmented = self.transform(

            image=eo,

            sar=np.stack([sar], axis=-1),

            mask=mask
        )

        eo_t = augmented["image"].float()

        sar_t = augmented["sar"].float()

        mask_t = augmented["mask"]

        # =========================
        # EO NORMALIZATION
        # =========================

        eo_t = eo_t / 255.0

        eo_mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        eo_std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        eo_t = (eo_t - eo_mean) / eo_std

        # =========================
        # SAR NORMALIZATION
        # =========================

        sar_mean = sar_t.mean()

        sar_std = sar_t.std() + 1e-8

        sar_t = (sar_t - sar_mean) / sar_std

        # Final 4-channel tensor
        image = torch.cat(
            [eo_t, sar_t],
            dim=0
        )

        return image, mask_t.long()


if __name__ == "__main__":
    dataset = ChangeDetectionDataset(
        root_dir=r"C:\Sashank\AIML\GalaxEye\data",
        split="train",
        image_size=256
    )

    image, mask = dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Mask shape:  {mask.shape}")
    print(f"Unique mask values: {mask.unique()}")
    print(f"Change ratio: {(mask == 1).float().mean().item()*100:.2f}%")