import h5py
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


class PatchGeneDataset(Dataset):
    def __init__(self, patch_file, gene_file, transform=None):
        """
        Args:
            patch_file (str): Path to the patch.h5 file.
            gene_file (str): Path to the gene.h5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patch_file = patch_file
        self.gene_file = gene_file
        self.transform = transform

        # Load h5 files
        with h5py.File(self.patch_file, 'r') as f:
            self.patches = f['img'][:]  # Assumes the data is stored under the 'patches' dataset
        with h5py.File(self.gene_file, 'r') as f:
            self.genes = f['genes'][:]  # Assumes the data is stored under the 'genes' dataset

        assert len(self.patches) == len(self.genes), "Patch and gene data lengths do not match."

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        return:
            patch: (3, 224, 224) float32 tensor
            gene: () float32 tensor with log1p normalized
        """
        patch = self.patches[idx]
        gene = self.genes[idx]

        # Apply transformations if any
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = torch.as_tensor(patch, dtype=torch.float32)  # [224, 224, 3]
            patch = patch.permute(2, 0, 1)  # [3, 224, 224] 0~255

        # Normalize to [-1, 1]
        # patch = patch / 127.5 - 1  # -1~1

        # Convert to torch tensors
        gene = torch.as_tensor(gene, dtype=torch.float32).squeeze()

        return patch, gene


def get_loader(config):
    # load data
    train_dataset = PatchGeneDataset(config["train_patch_file"], config["train_gene_file"])
    valid_dataset = PatchGeneDataset(config["valid_patch_file"], config["valid_gene_file"])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, valid_loader

def get_test_loader(config):
    # load test data
    dataset = PatchGeneDataset(config["test_patch_file"], config["test_gene_file"])
    # DataLoader
    test_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return test_loader


if __name__ == "__main__":
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = PatchGeneDataset(config["train_patch_file"], config["train_gene_file"])

    # get data
    patch, gene = dataset[0]
    print(patch.shape)  # torch.Size([3, 224, 224])
    print(gene.shape)   # torch.Size([100])
    print(patch.min(), patch.max())  # [-1, 1]

    train_loader, valid_loader = get_loader(config)
    for batch_idx, (patches, genes) in enumerate(train_loader):
        assert patches.shape == (config["batch_size"], 3, 224, 224), f"Train batch shape mismatch! Got {patches.shape}"
        assert genes.shape == (config["batch_size"], 100), f"Gene batch shape mismatch! Got {genes.shape}"
        break
    print("Train and validation loaders passed!")

    test_loader = get_test_loader(config)
    for batch_idx, (patches, genes) in enumerate(test_loader):
        assert patches.shape == (config["batch_size"], 3, 224, 224), f"Test batch shape mismatch! Got {patches.shape}"
        assert genes.shape == (config["batch_size"], 100), f"Gene batch shape mismatch! Got {genes.shape}"
        break
    print("Test loader passed!")

