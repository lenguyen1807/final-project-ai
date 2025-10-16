import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image

from src.dataset import (
    DatasetCollator,
    XRayDataset,
)

TEST_DIR = "test_xray_dataset"
CSV_PATH = os.path.join(TEST_DIR, "dataset.csv")


def setup_dummy_data():
    os.makedirs(TEST_DIR, exist_ok=True)

    # Create dummy grayscale (1x32x32) and RGB (3x32x32) images
    img1 = torch.rand(1, 2024, 2048)  # grayscale
    img2 = torch.rand(3, 2048, 2024)  # RGB

    img1_path = os.path.join(TEST_DIR, "img1.png")
    img2_path = os.path.join(TEST_DIR, "img2.png")

    save_image(img1, img1_path)
    save_image(img2, img2_path)

    # Create CSV
    df = pd.DataFrame(
        {"imgs": [img1_path, img2_path], "captions": ["grayscale image", "rgb image"]}
    )
    df.to_csv(CSV_PATH, index=False)


def test_dataset_and_collator():
    setup_dummy_data()

    data_transforms = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = XRayDataset(CSV_PATH, transform=data_transforms)
    collator = DatasetCollator()

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        print("Batch images shape:", batch["images"].shape)  # (B, C, H, W)
        print("Batch reports:", batch["reports"])
        assert batch["images"].ndim == 4
        assert len(batch["reports"]) == batch["images"].shape[0]


if __name__ == "__main__":
    test_dataset_and_collator()
