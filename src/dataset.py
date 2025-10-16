from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.tv_tensors import TVTensor


class XRayDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        super().__init__()

        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified CSV file was not found: {csv_path}")

        if "imgs" not in self.df.columns or "captions" not in self.df.columns:
            raise ValueError(
                f"CSV file at {csv_path} must contain 'imgs' and 'captions' columns."
            )

        self.transform = transform
        print(f"INFO: Loaded dataset from {csv_path} with {len(self.df)} samples.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = row["imgs"]
        caption = row["captions"]

        try:
            image_tensor = decode_image(image_path, mode=ImageReadMode.UNCHANGED)
        except FileNotFoundError:
            print(
                f"WARNING: Image file not found at {image_path}. Skipping sample at index {index}."
            )
            return self.__getitem__((index + 1) % len(self))
        except Exception as e:
            print(
                f"WARNING: Error loading image {image_path}: {e}. Skipping sample at index {index}."
            )
            return self.__getitem__((index + 1) % len(self))

        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.expand(3, -1, -1)

        tv_image = TVTensor(image_tensor)
        print(tv_image.shape)

        if self.transform:
            tv_image = self.transform(tv_image)
        print(tv_image.shape)

        if not isinstance(caption, str):
            caption = str(caption)

        return tv_image, caption


class DatasetCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[tuple]) -> Dict[str, Any]:
        if not batch:
            return {}
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return {}

        images, captions = zip(*batch)
        print([img.shape for img in images])
        batched_images = torch.stack(images, dim=0)
        return {"images": batched_images, "reports": list(captions)}
