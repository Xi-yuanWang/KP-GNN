from typing import Callable
from impl import EGNN
from mydataset import load_splited_dataset
import argparse
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch_geometric.data import Batch, Data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--task", type=int, default=None)
parser.add_argument("--nodetask", action="store_true")
parser.add_argument("--K", type=int, default=6)
args = parser.parse_args()
print(args)

dataargs = {"nodetask": args.nodetask}
if args.task != None:
    dataargs["task"] = args.task

device = torch.device("cuda")

trn_ds, val_ds, tst_ds = load_splited_dataset(args.dataset, args.K, dataargs)

def buildModel(hid_dim: int, out_dim: int, layer: int, dropout: float,
               pool: str):
    return EGNN.GNN(trn_ds[0].x.shape[-1],
                    args.K,
                    hid_dim,
                    layer,
                    dropout,
                    None,
                    out_dim=out_dim,
                    pool=pool)


def train(mod: nn.Module, opt: Optimizer, dl: DataLoader, loss_fn: Callable):
    mod.train()
    losss = []
    for data in dl:
        opt.zero_grad()
        data = data.to(device, non_blocking=True)
        pred = mod(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(pred.flatten(), data.y.flatten())
        loss.backward()
        losss.append(loss.item())
        opt.step()
    return np.average(losss)

@torch.no_grad()
def test(mod: nn.Module, dl: DataLoader, score_fn: Callable):
    mod.eval()
    preds = []
    ys = []
    for data in dl:
        data = data.to(device, non_blocking=True)
        pred = mod(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(pred.flatten().cpu())
        ys.append(data.flatten().y.cpu())
    pred = torch.cat(preds)
    y = torch.cat(ys)
    return score_fn(pred, y)

def work(hid_dim: int, num_layer:int, dropout:float, lr: float, batch_size: float, pool: str, max_epoch: int=250):
    mod = buildModel(hid_dim, 1, num_layer,dropout, pool).to(device)
    opt = Adam(mod.parameters(), lr)
    trn_dl = DataLoader(trn_ds, batch_size, True, drop_last=True)
    val_dl = DataLoader(val_ds, 512, False)
    tst_dl = DataLoader(tst_ds, 512, False)
    lowest_val = 1
    tst_score = 1
    for i in range(max_epoch):
        loss = train(mod, opt, trn_dl, F.mse_loss)
        val_score = train(mod, opt, val_dl, F.l1_loss)
        if val_score<lowest_val:
            lowest_val = val_score
            tst_score = train(mod, opt, tst_dl, F.l1_loss)
            print(f"epoch {i} trn {loss:.3e} val {val_score:.3e} tst {tst_score:.3e}")
        else:
            print(f"epoch {i} trn {loss:.3e} val {val_score:.3e} tst {tst_score:.3e}")
    return tst_score

import optuna
def opt(trial: optuna.Trial):
    hid_dim = trial.suggest_int("hid_dim", 32, 256, step=32)
    num_layer = trial.suggest_int("num_layer", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.9, step = 0.05)
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    pool = trial.suggest_categorical("pool", ["max", "sum", "mean"])
    return work(hid_dim, num_layer, dropout, lr, batch_size, pool)

stu = optuna.create_study(f"sqlite:///opt/{args.dataset}_{args.task}.db", study_name=f"{args.dataset}_{args.task}", load_if_exists=True, direction="minimize")
stu.optimize(opt, 400)