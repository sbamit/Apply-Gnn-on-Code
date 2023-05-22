import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch GraphConv
import dgl
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
# import helpers.losses as svdloss
import helpers.ml as ml
import helpers.rank_eval as svdr
import helpers.gnnexplainer as lvdgne


class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512,
        embtype: str = "codebert",
        embfeat: int = -1,  # Keep for legacy purposes
        num_heads: int = 4,
        lr: float = 1e-3,
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce",
        multitask: str = "linemethod",
        stmtweight: int = 5,
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()

        # Set params based on embedding type
        # if self.hparams.embtype == "codebert":
        self.hparams.embfeat = 768
        self.EMBED = "_CODEBERT"

        # Loss
        self.loss = th.nn.CrossEntropyLoss(
            weight=th.Tensor([1, self.hparams.stmtweight]).cpu()
        )
        self.loss_f = th.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy('binary')
        self.auroc = torchmetrics.AUROC('binary', compute_on_step=False)
        self.mcc = torchmetrics.MatthewsCorrCoef(task='binary')

        # GraphConv Type
        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        # if self.hparams.gnntype == "gat":
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        """elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}"""

        # model: gat2layer
        # if "gat" in self.hparams.model:
        self.gat = gnn(**gnn1_args)
        self.gat2 = gnn(**gnn2_args)
        fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
        self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
        self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
        self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # Transform codebert embedding
        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        # Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass.

        data = BigVulDatasetLineVDDataModule(batch_size=1, sample=2, nsampling=True)
        g = next(iter(data.train_dataloader()))

        e_weights and h_override are just used for GNNExplainer.
        """
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            # h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
        else:
            g2 = g
            h = g.ndata[self.EMBED]
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            # h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device) 
            # th.rand(h_func.shape[0], 2).to(self.device)

        # Transform h_func if wrong size
        """if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)"""

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            # h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            # h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        # h_func = self.fc2(h_func)  
        # Share weights between method-level and statement-level tasks

        """if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:"""
        return h, None  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_LABEL").long()
            # labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_LABEL"].long()
                # labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_LABEL"].long()
                # labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, None

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(
            batch
        )  # Labels func should be the method-level label for statements
        # print(logits.argmax(1), labels_func)
        loss1 = self.loss(logits[0], labels)
        """if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)"""
        # Need some way of combining the losses for multitask training
        loss = 0
        if "line" in self.hparams.multitask:
            loss1 = self.loss(logits[0], labels)
            loss += loss1
        """if "method" in self.hparams.multitask and not self.hparams.methodlevel:
            loss2 = self.loss(logits[1], labels_func)
            loss += loss2"""

        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        """if not self.hparams.methodlevel:
            acc_func = self.accuracy(logits.argmax(1), labels_func)"""
        mcc = self.mcc(pred.argmax(1), labels)
        print("\nTraining Batch", batch_idx, "\t", "train_loss", loss.detach().numpy(), "\t", "train_acc", acc.numpy())
        # print("train_loss", loss, end="\t")
        # print("train_acc", acc, end="\t")
        print("train_mcc", mcc, end="\t")

        # self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_acc", acc, prog_bar=True, logger=True)
        """if not self.hparams.methodlevel:
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True)"""
        # self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss = 0
        if "line" in self.hparams.multitask:
            loss1 = self.loss(logits[0], labels)
            loss += loss1
        """if "method" in self.hparams.multitask:
            loss2 = self.loss_f(logits[1], labels_func)
            loss += loss2"""

        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        print("\nValidating Batch", batch_idx, "\t", "val_acc", acc.numpy())
        # self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logits[:, 1], labels)
        # self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        # self.log("val_acc", acc, prog_bar=True, logger=True)
        # self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, _ = self.shared_step(
            batch, True
        )  # TODO: Make work for multitask

        if self.hparams.methodlevel:
            labels_f = labels
            return logits[0], labels_f, dgl.unbatch(batch)

        batch.ndata["pred"] = F.softmax(logits[0], dim=1)
        # batch.ndata["pred_func"] = F.softmax(logits[1], dim=1)
        # logits_f = []
        # labels_f = []
        preds = []
        for i in dgl.unbatch(batch):
            preds.append(
                [
                    list(i.ndata["pred"].detach().cpu().numpy()),
                    list(i.ndata["_LABEL"].detach().cpu().numpy()),
                    # i.ndata["pred_func"].argmax(1).detach().cpu(),
                    list(i.ndata["_LINE"].detach().cpu().numpy()),
                ]
            )
        loss = self.loss(logits[0], labels)
        print("\nTesting Batch", batch_idx, '\tLoss',  loss.detach().numpy())
        # logits_f.append(dgl.mean_nodes(i, "pred_func").detach().cpu())
        # labels_f.append(dgl.mean_nodes(i, "_FVULN").detach().cpu())
        # return [logits[0], logits_f], [labels, labels_f], preds
        return logits[0], labels, preds

    def plot_pr_curve(self):
        """Plot Precision-Recall Curve for Positive Class (after test)."""
        precision, recall, thresholds = precision_recall_curve(
            self.linevd_true, [i[1] for i in self.linevd_pred]
        )
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot()
        return

    def configure_optimizers(self):
        """Configure optimizer."""
        return th.optim.AdamW(self.parameters(), lr=self.lr)


def get_relevant_metrics(trial_result):
    """Get relevant metrics from results."""
    ret = {}
    ret["trial_id"] = trial_result[0]
    ret["checkpoint"] = trial_result[1]
    ret["acc@5"] = trial_result[2][5]
    ret["stmt_f1"] = trial_result[3]["f1"]
    ret["stmt_rec"] = trial_result[3]["rec"]
    ret["stmt_prec"] = trial_result[3]["prec"]
    ret["stmt_mcc"] = trial_result[3]["mcc"]
    ret["stmt_fpr"] = trial_result[3]["fpr"]
    ret["stmt_fnr"] = trial_result[3]["fnr"]
    ret["stmt_rocauc"] = trial_result[3]["roc_auc"]
    ret["stmt_prauc"] = trial_result[3]["pr_auc"]
    ret["stmt_prauc_pos"] = trial_result[3]["pr_auc_pos"]
    ret["func_f1"] = trial_result[4]["f1"]
    ret["func_rec"] = trial_result[4]["rec"]
    ret["func_prec"] = trial_result[4]["prec"]
    ret["func_mcc"] = trial_result[4]["mcc"]
    ret["func_fpr"] = trial_result[4]["fpr"]
    ret["func_fnr"] = trial_result[4]["fnr"]
    ret["func_rocauc"] = trial_result[4]["roc_auc"]
    ret["func_prauc"] = trial_result[4]["pr_auc"]
    ret["MAP@5"] = trial_result[5]["MAP@5"]
    ret["nDCG@5"] = trial_result[5]["nDCG@5"]
    ret["MFR"] = trial_result[5]["MFR"]
    ret["MAR"] = trial_result[5]["MAR"]
    ret["stmtline_f1"] = trial_result[6]["f1"]
    ret["stmtline_rec"] = trial_result[6]["rec"]
    ret["stmtline_prec"] = trial_result[6]["prec"]
    ret["stmtline_mcc"] = trial_result[6]["mcc"]
    ret["stmtline_fpr"] = trial_result[6]["fpr"]
    ret["stmtline_fnr"] = trial_result[6]["fnr"]
    ret["stmtline_rocauc"] = trial_result[6]["roc_auc"]
    ret["stmtline_prauc"] = trial_result[6]["pr_auc"]
    ret["stmtline_prauc_pos"] = trial_result[6]["pr_auc_pos"]

    ret = {k: round(v, 3) if isinstance(v, float) else v for k, v in ret.items()}
    ret["learning_rate"] = trial_result[7]
    ret["stmt_loss"] = trial_result[3]["loss"]
    ret["func_loss"] = trial_result[4]["loss"]
    ret["stmtline_loss"] = trial_result[6]["loss"]
    return ret