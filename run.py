# import linevd_gnn as lvd
import argparse
import os
import graph_dataset as gd
import graph_model as gm
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import (TuneReportCallback, TuneReportCheckpointCallback)


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
        batch_size=8)
    model = gm.LitGNN()
    # Train model
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    # metrics = ["train_loss", "val_loss", "val_auroc"]
    # raytune_callback = TuneReportCallback(metrics, on="validation_end")
    # rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    max_epochs = 10
    savepath = os.curdir
    # trainer = pl.Trainer(max_epochs=max_epochs)
    trainer = pl.Trainer(
        # gpus=1,
        # auto_lr_find=False,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        # callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)
    
    # test the model
    trainer.test(model, dataloaders=data.test_dataloader())

    """folder_list = [folder for folder in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, folder))]
    folder_list.remove("out") if "out" in folder_list else None
    os.chdir(master_dir)  # in the master directory
    # Test get_node_edges function
    # probably will call this function from outside
    pd.set_option('display.max_rows', None)
    for folder in folder_list:
        os.chdir(folder)
        file_list = [file for file in os.listdir() if file.endswith(".c")]
        for filename in file_list:
            nodes, _ = cnad.get_node_edges(filename)
            print(nodes.columns)
            nodes.fillna('<empty>')
            print(nodes[['id', '_label', 'name', 'code', 'typeFullName',
                        'controlStructureType', 'node_label']])
        os.chdir(os.pardir)
    # Go back to the Project direcotry
    os.chdir(proj_dir)"""


# proj_dir = os.getcwd()     
if __name__ == "__main__": 
    main()
