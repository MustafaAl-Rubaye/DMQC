import hydra
import gc
import torch
import numpy as np
import random
import pytorch_lightning as pl
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import train_test_split
from dmqc.pipeline.lightning import Mylightningclass, my_transforms
from dmqc.pipeline.helper import path_to_df


@hydra.main(config_path="../../conf/config.yaml")
def main(cfg):
    print(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    df = path_to_df(cfg)

    dataframe_train_val, dataframe_test = train_test_split(
        df, test_size=0.2, random_state=cfg.seed,
    )

    kfold = GroupKFold(n_splits=cfg.n_folds)

    transforms = my_transforms()

    for train, val in kfold.split(
        dataframe_train_val, groups=dataframe_train_val["id"]
    ):

        print(cfg.backbone)
        print(cfg.model)
        train_df = dataframe_train_val.iloc[train]
        val_df = dataframe_train_val.iloc[val]

        my_class = Mylightningclass(
            val_df=val_df,
            train_df=train_df,
            val_stream=transforms["eval"],
            train_stream=transforms["train"],
            cfg=cfg,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.n_epoches,
            distributed_backend=cfg.gpus.backend,
            gpus=list(cfg.gpus.ids),
        )

        trainer.fit(my_class)
        del my_class
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        # trainer.test(ckpt_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/vgg11_bn_fpn/lightning_logs/version_0/checkpoints/epoch=169.ckpt")
        # trainer.test(ckpt_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/vgg11_bn_fpn/lightning_logs/version_1/checkpoints/epoch=179.ckpt")
        # trainer.test(ckpt_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/vgg11_bn_fpn/lightning_logs/version_2/checkpoints/epoch=101.ckpt")
        # trainer.test(ckpt_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/vgg11_bn_fpn/lightning_logs/version_3/checkpoints/epoch=280.ckpt")
        # trainer.test(ckpt_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/multirun/6_mod_300_epo/best_mode/vgg11_bn_fpn/lightning_logs/version_4/checkpoints/epoch=162.ckpt")


if __name__ == "__main__":
    main()
