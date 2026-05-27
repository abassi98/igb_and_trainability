
"""Dataset wrappers used by the experiment entry points."""

from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    """Adapter around the Hugging Face Tiny ImageNet dataset.

    The wrapper exposes a torchvision-style dataset interface and adds a
    ``targets`` attribute so downstream code can interact with it like a standard
    classification dataset.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        # This line creates a .targets attribute as a list of int labels
        self.targets = [item['label'] for item in hf_dataset]

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]['image']
        img = img.convert("RGB") # force to have three channels
        label = self.hf_dataset[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.hf_dataset)
