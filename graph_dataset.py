"""Main code for training. Probably needs refactoring."""
import os
import dgl
import pandas as pd
import pytorch_lightning as pl
import torch as th
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import codebert as cb
import create_nodes_and_edges as svdj
from dgl.dataloading import GraphDataLoader
# from dgl.data.utils import load_graphs, save_graphs
# from torchmetrics import MatthewsCorrCoef


def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """Extract graph feature (basic).

    _id = svddc.BigVulDataset.itempath(177775)
    _id = svddc.BigVulDataset.itempath(180189)
    _id = svddc.BigVulDataset.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    # n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        try:
            func_name = _id.split('.')[0] #n[n.lineNumber == 1].name.item()
        except Exception as E:
            print(_id, E)
            func_name = ""
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
    else:
        n.code = "</s>" + " " + n.code

    ntypes = n.typeFullName.tolist()
    # Return plain-text code, line number list, innodes, outnodes
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), ntypes, etypes


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i+n]


# Class that implements LightningDataModule
class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        batch_size: int = 8,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
        master_dir: str = "test",
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        # dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        codebert = cb.CodeBert()
        folder_list = [folder for folder in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, folder))]
        folder_list.remove("out") if "out" in folder_list else None
        os.chdir(master_dir)  # in the master directory
        # Test get_node_edges function
        # probably will call this function from outside
        self.g_list = []
        pd.set_option('display.max_rows', None)
        for folder in tqdm(folder_list):
            os.chdir(folder)
            file_list = [file for file in os.listdir() if file.endswith(".c")]
            for filename in file_list:
                code, lineno, ei, eo, ntypes, etypes = feature_extraction(filename)
                label = [1 if (n == 'int' or n == 'double' or n == 'float' or n == 'uint' or n == 'long')
                         else 0 for n in ntypes]
                
                g = dgl.graph((eo, ei))
                code = [c.replace("\\t", "").replace("\\n", "") for c in code]
                chunked_batches = chunks(code, 128)
                features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
                g.ndata["_CODEBERT"] = th.cat(features)
                g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
                g.ndata["_LINE"] = th.Tensor(lineno).int()
                g.ndata["_TYPE"] = th.Tensor(ntypes).int()
                g.edata["_ETYPE"] = th.Tensor(etypes).int()
                
                g.ndata["_LABEL"] = th.Tensor(label).int()
                g = dgl.add_self_loop(g)
                # save_graphs(str(savedir), [g])
                self.g_list.append(g)
                # print(g)
            os.chdir(os.pardir)
        # Go back to the Project direcotry
        os.chdir(svdj.proj_dir)
      
        # Train Test Split
        self.train, self.test = train_test_split(self.g_list, test_size=0.1, shuffle=True)
        self.train, self.val = train_test_split(self.train, test_size=0.1, shuffle=True)
        print("\n\ng_train:", self.train)
        print("\n\ng_test:", self.test)
        print("\n\ng_val:", self.val, "\n\n")

        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops

    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.NodeDataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=1,
        )

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size)
        
    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val))))
            return self.node_dl(g)
        return GraphDataLoader(self.val, batch_size=self.batch_size)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, batch_size=32)
