import os
import logging
import time

# import sys
# import argparse
# import numpy as np

import models.esrgan.options.options as option
import models.esrgan.utils.util as util
from models.esrgan.data import create_dataset, create_dataloader
from models.esrgan.models import create_model


# parser = argparse.ArgumentParser()
# parser.add_argument("-opt", type=str, required=True, help="Path to options JSON file.")


def test():
    # options
    jsonPath = "models/esrgan/options/test_ESRGAN.json"
    opt = option.parse(jsonPath, is_train=False)
    util.mkdirs(
        (path for key, path in opt["path"].items() if not key == "pretrain_model_G")
    )
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(
        None, opt["path"]["log"], "test.log", level=logging.INFO, screen=True
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info(
            "Number of test images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(test_set)
            )
        )
        test_loaders.append(test_loader)

    # Create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        logger.info("\nTesting [{:s}]...".format(test_set_name))
        # dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)

        # where to save sr img
        dataset_dir = "img/sr/esrgan"
        util.mkdir(dataset_dir)

        for data in test_loader:
            need_HR = False if test_loader.dataset.opt["dataroot_HR"] is None else True

            model.feed_data(data, need_HR=need_HR)
            img_path = data["LR_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals["SR"])  # uint8

            # save images
            suffix = opt["suffix"]
            if suffix:
                save_img_path = os.path.join(
                    dataset_dir,
                    img_name + suffix + os.path.splitext(os.path.basename(img_path))[1],
                )
            else:
                save_img_path = os.path.join(dataset_dir, os.path.basename(img_path))
            util.save_img(sr_img, save_img_path)

            logger.info(img_name)
