# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/28/2023
# ============================================================================

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

from pkg_manager import *
from para_manager import *

def csdiTrain(cfWindowSize, model, config, train_loader, valid_loader=None, valid_epoch_interval = 1, foldername="",):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + multiTimeSeries + '_' + str(vae_batch_size) + '_' +str(cfWindowSize) + '_' + X + "_csdi_model.pth"


    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (observedTp, observedData, observedMask, groundTruthMask) in enumerate(it, start = 1):
                optimizer.zero_grad()

                loss = model(observedTp.to(device), observedData.to(device), observedMask.to(device), groundTruthMask.to(device))
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no,},refresh=False,)
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, (observedTp, observedData, observedMask, groundTruthMask) in enumerate(it, start=1):
                        loss = model(observedTp.to(device), observedData.to(device), observedMask.to(device), groundTruthMask.to(device), is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(ordered_dict={"valid_avg_epoch_loss": avg_loss_valid / batch_no,"epoch": epoch_no,},refresh=False,)

            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at",epoch_no,)
                if foldername != "":
                    torch.save(model, output_path)
