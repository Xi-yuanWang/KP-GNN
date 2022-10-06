from impl.preprocess import multihopkernel
from datasets.ZINC_dataset import ZINC
from functools import partial
import subprocess


def load_splited_dataset(name: str, K: int, args: dict={}):
    path=f"data/{name}"
    pret = partial(multihopkernel, K=K)
    subprocess.call("rm -rf data/*/processed", shell=True)
    subprocess.call("rm -rf data/*/*/processed", shell=True)
    if name == "ZINC":
        path="data/" + "ZINC"
        trainset = ZINC(path, subset=True, split="train", pre_transform=pret)
        valset = ZINC(path, subset=True, split="val", pre_transform=pret)
        testset = ZINC(path, subset=True, split="test", pre_transform=pret)
        return trainset, valset, testset
    elif name == "subgraphcount":
        from datasets.GraphCountDataset import GraphCountDataset
        dataset = GraphCountDataset(path, pre_transform=pret)
        dataset.data.y=dataset.data.y/dataset.data.y.std(0)
        train_dataset, val_dataset, test_dataset = dataset[dataset.train_idx], dataset[dataset.val_idx], dataset[dataset.test_idx]
        return train_dataset, val_dataset, test_dataset
    elif name == "pna-simulation":
        from datasets.GraphPropertyDataset import GraphPropertyDataset
        train_dataset = GraphPropertyDataset(path, split='train', pre_transform=pret)
        val_dataset = GraphPropertyDataset(path, split='val', pre_transform=pret)
        test_dataset = GraphPropertyDataset(path, split='test', pre_transform=pret)
        train_dataset = train_dataset[train_dataset.indices()]
        val_dataset = val_dataset[val_dataset.indices()]
        test_dataset = test_dataset[test_dataset.indices()]
        task = args["task"]
        if args["nodetask"]:
            train_dataset.data.y = train_dataset.data.pos[:,task:task+1]
            val_dataset.data.y = val_dataset.data.pos[:, task: task+1]
            test_dataset.data.y = test_dataset.data.pos[:, task: task+1]
        else:
            train_dataset.data.y = train_dataset.data.y[:,task:task+1]
            val_dataset.data.y = val_dataset.data.y[:, task: task+1]
            test_dataset.data.y = test_dataset.data.y[:, task: task+1]
        return train_dataset, val_dataset, test_dataset
    # elif name == "qm9"
    # elif name == "ogb-mol"
    else:
        raise NotImplementedError("dataset with edge_attr are not implemented")

def load_dataset(name: str):
    if name in ['MUTAG',"DD", 'PROTEINS',"PTC","IMDBBINARY"]:
        from datasets.tu_dataset import TUDatasetGINSplit,TUDataset
        path="data/TUGIN_"+name
        path=path
        if name=="DD":
            # NGNN setting all use this
            dataset = TUDataset(path, name, cleaned=False, pre_transform=multihopkernel) 
        else:
            dataset = TUDatasetGINSplit(name ,path, pre_transform=multihopkernel)
    