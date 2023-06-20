"""Main code for training. Probably needs refactoring."""
import os
from pathlib import Path
import dgl
import pandas as pd
import pytorch_lightning as pl
import torch as th
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import codebert as cb
import create_nodes_and_edges as cnad
from dgl.dataloading import GraphDataLoader
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
    nl = cnad.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """Extract graph feature (basic).
    return_nodes arg is used to get the node information.
    """
    # Get CPG
    n, e = cnad.get_node_edges(_id)
    # n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = cnad.rdg(e, graph_type.split("+")[0])
    n = cnad.drop_lone_nodes(n, e)

    # Plot graph
    # cnad.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Don't Append function name to code
    """if "+raw" not in graph_type:
        try:
            func_name = _id.split('.')[0] #n[n.lineNumber == 1].name.item()
        except Exception as E:
            print(_id, E)
            func_name = ""
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
    else:
        n.code = "</s>" + " " + n.code"""

    # Prepare attributes to return
    ntypes = n.typeFullName.tolist()
    labels = n._label.tolist()
    codes = n.code.tolist()
    node_ids = n.id.tolist()
    # line_nums = n.lineNumber.tolist() 
    # node_labels = n.node_label.tolist()
    in_nodes = e.innode.tolist()
    out_nodes = e.outnode.tolist()
    # Return plain-text code, line number list, innodes, outnodes
    return codes, node_ids, in_nodes, out_nodes, ntypes, etypes, labels


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i+n]


# Class that implements LightningDataModule
class DglGraphDataset(pl.LightningDataModule):
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
        self.codebert = cb.CodeBert()
        folder_list = [folder for folder in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, folder))]
        folder_list.remove("out") if "out" in folder_list else None
        os.chdir(master_dir)  # in the master directory
        # Test get_node_edges function
        # probably will call this function from outside
        self.g_list = []
        for folder in tqdm(folder_list):
            os.chdir(folder)
            file_list = [file for file in os.listdir() if file.endswith(".c")]
            for filename in file_list:
                g = self.create_or_load_graph(filename)
                self.g_list.append(g)
            os.chdir(os.pardir)
        # Go back to the Project direcotry
        os.chdir(cnad.proj_dir)
      
        # Train Test Split
        self.train, self.test = train_test_split(self.g_list, test_size=0.1, shuffle=True)
        self.train, self.val = train_test_split(self.train, test_size=0.1, shuffle=True)

        # Print # of datapoints
        print("\nDataset Summary")
        print("# of training graphs", len(self.train))
        print("# of validation graphs", len(self.val))
        print("# of testing graphs", len(self.test))
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops

    def create_or_load_graph(self, filename):
        # print("\nWorking on", filename)
        savedir = str(Path(filename)) + ".bin"
        # print("savedir", savedir)
        if os.path.exists(savedir):
            g = dgl.load_graphs(str(savedir))[0][0]
            return g
        class_labels = []
        code, node_ids, ei, eo, ntypes, etypes, labels = feature_extraction(filename)
        # Create Labels for Classification
        for i in range(len(node_ids)):
            node_type = ntypes[i]
            _label = labels[i]
            if (_label == 'IDENTIFIER' and node_type.find('undefined') != -1):
                class_labels.append(0)
            elif (_label == 'IDENTIFIER' and node_type == 'ANY'):
                class_labels.append(0)
            else:
                class_labels.append(1)
        
        g = dgl.graph((eo, ei))
        # printGraph(filename, g)
        # printFeatures(code, node_ids, ei, eo, ntypes, etypes, labels, class_labels)
        code = [c.replace("\\t", "").replace("\\n", "") for c in code]
        chunked_batches = chunks(code, 128)
        features = [self.codebert.encode(c).detach().cpu() for c in chunked_batches]
        g.ndata["_CODEBERT"] = th.cat(features)
        g.ndata["_IDS"] = th.Tensor(node_ids).int()
        g.ndata["_TYPE"] = th.Tensor(ntypes).int()
        g.edata["_ETYPE"] = th.Tensor(etypes).int()
        
        g.ndata["_LABEL"] = th.Tensor(class_labels).int()
        g = dgl.add_self_loop(g)
        dgl.save_graphs(str(savedir), [g])
        return g

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
        return GraphDataLoader(self.train, shuffle=True, 
                               batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return GraphDataLoader(self.val, batch_size=self.batch_size)


def printGraph(file, g):
    print('\nfor ', file,'->', g)
    # print(g.ndata['_LABEL'])


def printFeatures(code, node_ids, ei, eo, ntypes, etypes, labels, class_labels):
    pd.set_option('display.max_rows', None)
    nodes_df = pd.DataFrame(data=[code, node_ids, ntypes, labels, class_labels]).transpose()
    nodes_df.columns = ['code', 'ID', 'typeFullName', '_label', 'class_label']
    nodes_df.set_index('ID')
    print('Nodes\n', nodes_df)

    """edges_df = pd.DataFrame(data=[ei, eo, etypes]).transpose()
    edges_df.columns = ['ei', 'eo', 'etypes']
    print('Edges\n', edges_df)
    """
