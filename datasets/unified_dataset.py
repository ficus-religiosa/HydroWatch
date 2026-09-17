import torch
from torch.utils.data import Dataset


class UnifiedMarineDebrisDataset(Dataset):
    """
    Concatenates the three dataset adapters.

    Each adapter must return:
        image
        boxes
        labels
        mask
        image_id
        dataset

    No class mapping is invented here. The three source
    vocabularies must be explicitly mapped before training.
    """

    def __init__(
        self,
        datasets
    ):
        if not datasets:
            raise ValueError(
                "At least one dataset is required."
            )

        self.datasets = list(datasets)

        self.offsets = []
        total = 0

        for ds in self.datasets:
            self.offsets.append(total)
            total += len(ds)

        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        for ds, offset in reversed(
            list(zip(
                self.datasets,
                self.offsets
            ))
        ):
            if index >= offset:
                return ds[index - offset]

        raise IndexError(index)


def collate_detection_batch(batch):
    images = torch.stack(
        [x["image"] for x in batch]
    )

    return {
        "images": images,
        "targets": [
            {
                "boxes": x["boxes"],
                "labels": x["labels"],
                "mask": x["mask"],
                "image_id": x["image_id"],
                "dataset": x["dataset"]
            }
            for x in batch
        ]
    }
