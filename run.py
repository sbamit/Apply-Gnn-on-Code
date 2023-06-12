# import linevd_gnn as lvd
import argparse
import os
import graph_dataset as gd
import graph_model_simpler as gm_simple
import pytorch_lightning as pl
import warnings
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--master_dir", help="This directory contains data")
    args = parser.parse_args()
    master_dir = ""
    # check if anythign was passed in
    if args.master_dir is None:
        print("[*] The master directory was not specified, try again")
        exit()
    else:
        master_dir = args.master_dir
        print(f"[*] Master Directory set as {master_dir}")

    # Load data and model
    data = gd.DglGraphDataset(
        master_dir=master_dir,
        batch_size=16)
    model = gm_simple.LitGNN()
    print("\n\nModel Summary\n", model)
    # Train model
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    # metrics = ["train_loss", "val_loss", "val_auroc"]
    # raytune_callback = TuneReportCallback(metrics, on="validation_end")
    # rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    max_epochs = 10
    savepath = os.curdir
    trainer = pl.Trainer(
        # gpus=1,
        # auto_lr_find=False,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        # callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)

    # Load Model using Checkpoints
    # chk_path = "./lightning_logs/version_39/checkpoints/epoch=9-step=2060.ckpt"
    # model = gm_simple.LitGNN().load_from_checkpoint(chk_path)
    # test the model
    trainer.test(model, dataloaders=data.test_dataloader())


if __name__ == "__main__": 
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*smaller than the logging interval.*")
    warnings.filterwarnings("ignore", ".*TypedStorage is deprecated.*")
    main()
