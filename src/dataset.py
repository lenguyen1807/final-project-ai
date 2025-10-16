from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from torchvision.tv_tensors import TVTensor


class KaggleChestXRayDataset(Dataset):
    """
    A PyTorch Dataset for the Kaggle Chest X-ray dataset format.
    Reads a CSV file with 'imgs' and 'captions' columns.

    Args:
        csv_path (str): The path to the train or validation CSV file.
        transform (callable, optional): A function/transform to be applied to the image.
    """

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        super().__init__()

        # --- Input Validation ---
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
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - The transformed image as a torch.Tensor.
            - The corresponding caption as a string.
        """
        # --- Data Retrieval ---
        row = self.df.iloc[index]
        image_path = row["imgs"]
        caption = row["captions"]

        # --- Image Loading and Preprocessing ---
        try:
            # Open the image using Pillow
            # CRITICAL: Convert grayscale X-ray to 3-channel RGB as the ViT expects 3 channels.
            image_tensor = decode_image(image_path, mode=ImageReadMode.UNCHANGED)
        except FileNotFoundError:
            print(
                f"WARNING: Image file not found at {image_path}. Skipping sample at index {index}."
            )
            # To handle this gracefully, we can return the next valid sample
            return self.__getitem__((index + 1) % len(self))
        except Exception as e:
            print(
                f"WARNING: Error loading image {image_path}: {e}. Skipping sample at index {index}."
            )
            return self.__getitem__((index + 1) % len(self))

        # Apply transformations if they are provided
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.expand(3, -1, -1)

        tv_image = TVTensor(image_tensor)
        if self.transform:
            tv_image = self.transform(tv_image)

        # Ensure the caption is a string, handle potential NaN values from pandas
        if not isinstance(caption, str):
            caption = str(caption)

        return tv_image, caption


class KaggleChestXRayCollator:
    """
    A collator that handles both image resizing and report tokenization.
    """

    def __init__(self, tokenizer: Callable, image_size: int, max_txt_len: int = 100):
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        # Create a transform to resize images to a consistent size and normalize them.
        self.image_transform = v2.Compose(
            [
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, batch: List[tuple]) -> Dict[str, Any]:
        if not batch:
            return {}

        # Filter out any None samples that might have resulted from loading errors
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return {}

        images, captions = zip(*batch)

        # --- Image Processing ---
        # Apply the transform to each image in the batch.
        processed_images = [self.image_transform(img) for img in images]
        batched_images = torch.stack(processed_images, dim=0)

        # --- Text Processing ---
        # Tokenize the captions.
        text_tokens = self.tokenizer(
            list(captions),
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )

        return {
            "images": batched_images,
            "reports": list(captions),
            "text_tokens": text_tokens,
        }
