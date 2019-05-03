"""create dataset and dataloader"""
import logging
import torch.utils.data

from models.esrgan.data.LRHR_dataset import LRHRDataset as D


def create_dataloader(dataset, dataset_opt):
    """create dataloader """
    phase = dataset_opt["phase"]
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["use_shuffle"],
            num_workers=dataset_opt["n_workers"],
            drop_last=True,
            pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )


def create_dataset(dataset_opt):
    """create dataset"""
    dataset = D(dataset_opt)
    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
