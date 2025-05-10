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

    train_df, val_df = train_test_split(
        dataframe_train_val, test_size=0.2, random_state=cfg.seed,
    )

    transforms = my_transforms()

    # print(cfg.backbone)
    # print(cfg.model)

    model = Mylightningclass.load_from_checkpoint(
        checkpoint_path="/home/mustafa/Documents/DMQC/dmqc/pipeline/outputs/2020-09-07/17-39-08/lightning_logs/version_0/checkpoints/epoch=212.ckpt",
        hparams_file="/home/mustafa/Documents/DMQC/dmqc/pipeline/outputs/2020-09-07/17-39-08/lightning_logs/version_0/hparams.yaml",
        map_location=None,
        # val_df=val_df,
        train_df=train_df,
        test_df=dataframe_test,
        val_stream=transforms["eval"],
        train_stream=transforms["train"],
        cfg=cfg,
    )

    my_class = Mylightningclass(
        val_df=val_df,
        train_df=train_df,
        test_df=dataframe_test,
        val_stream=transforms["eval"],
        train_stream=transforms["train"],
        cfg=cfg,
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=cfg.n_epoches,
        distributed_backend=cfg.gpus.backend,
        gpus=list(cfg.gpus.ids),
    )

    # trainer.fit(my_class)
    trainer.test(model)
    del my_class
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
