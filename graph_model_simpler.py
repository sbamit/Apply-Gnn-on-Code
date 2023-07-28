import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.nn.pytorch import GraphConv
import dgl
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve


class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512,
        embfeat: int = -1,  # Keep for legacy purposes
        lr: float = 1e-6,
        hdropout: float = 0.2,
        stmtweight: int = 5,
        mlpdropout: float = 0.2,
        model: str = "gcn2layer",
        nsampling: bool = False,
        random: bool = False
    ):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # Set params based on embedding type
        self.hparams.embfeat = 768
        self.hparams.hfeat = 512
        self.hparams.hdropout = hdropout
        self.EMBED = "_CODEBERT"

        # Loss
        self.loss = th.nn.CrossEntropyLoss(
            weight=th.Tensor([1, self.hparams.stmtweight]).cpu()
        )

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary', compute_on_step=False)
        # self.mcc = torchmetrics.MatthewsCorrCoef(task='binary')

        # GraphConv Type
        hfeat = self.hparams.hfeat
        embfeat = self.hparams.embfeat
        # gnn = GraphConv
        # gnn1_args = {"in_feats": embfeat, "out_feats": hfeat}
        # gnn2_args = {"in_feats": hfeat, "out_feats": hfeat}

        # model: gcn2layer
        self.conv1 = GraphConv(in_feats=embfeat, out_feats=hfeat)  # gnn(**gnn1_args)
        self.conv2 = GraphConv(in_feats=hfeat, out_feats=hfeat)  # gnn(**gnn2_args)
        fcin = hfeat
        self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
        # Hidden Layers
        self.fch = []
        for _ in range(4):
            self.fch.append(th.nn.Linear(
                self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass."""
        g2 = g
        h = g.ndata[self.EMBED]
        if len(feat_override) > 0:
            h = g.ndata[feat_override]
        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device)

        # model: gcn2layer
        h = self.conv1(g, h)
        h = self.conv2(g2, h)
        h = self.hdropout(F.elu(self.fc(h)))
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
        h = self.fc2(h)

        return h, None  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        labels = batch.ndata["_LABEL"].long()
        return logits, labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels = self.shared_step(batch)
        loss = self.loss(logits[0], labels)
        logits = logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        # print("\nTraining Batch", batch_idx, "\t", "train_loss", loss.detach().numpy(), "\t", "train_acc", acc.numpy())
        batch_dictionary = {"train_loss": loss,
                            "train_acc": acc
                            }
        self.training_step_outputs.append(batch_dictionary)
        # self.log("train_loss", loss, batch_size=8, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        batch_losses = [x['train_loss'] for x in outputs]
        epoch_loss = th.stack(batch_losses).mean()   # Combine losses
        epoch_acc = sum([x['train_acc'] for x in outputs])/len(outputs)
        print("\nTraining accuracy:", epoch_acc.cpu().numpy(), end='\t')
        print("Training loss:", epoch_loss.detach().cpu().numpy(), end='\n\n')
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        logits, labels = self.shared_step(batch)
        loss = self.loss(logits[0], labels)
        logits = logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        batch_dictionary = {"val_loss": loss,
                            "val_acc": acc
                            }
        self.validation_step_outputs.append(batch_dictionary)
        self.auroc.update(logits[:, 1], labels)
        # self.log("val_loss", loss, on_step=True,batch_size=8, prog_bar=True, logger=True)
        # self.log("val_auroc", self.auroc, batch_size=8, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = th.stack(batch_losses).mean()   # Combine losses
        epoch_acc = sum([x['val_acc'] for x in outputs])/len(outputs)
        print("\nValidation accuracy : ", epoch_acc.cpu().numpy(), end='\t')
        print("Validation loss : ", epoch_loss.cpu().numpy(), end='\n\n')
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels = self.shared_step(
            batch, True
        )

        batch.ndata["pred"] = F.softmax(logits[0], dim=1)
        preds = []
        for i in dgl.unbatch(batch):
            preds.append(
                [
                    list(i.ndata["pred"].detach().cpu().numpy()),
                    list(i.ndata["_LABEL"].detach().cpu().numpy()),
                    list(i.ndata["_IDS"].detach().cpu().numpy()),
                ]
            )

        pred = F.softmax(logits[0], dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        loss = self.loss(logits[0], labels)
        batch_dictionary = {"test_loss": loss,
                            "test_acc": acc
                            }
        self.test_step_outputs.append(batch_dictionary)
        return logits[0], labels, preds

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = th.stack(batch_losses).mean()   # Combine losses
        epoch_acc = sum([x['test_acc'] for x in outputs])/len(outputs)
        print("\nTest accuracy : ", epoch_acc.cpu().numpy(), end='\t')
        print("Test loss : ", epoch_loss.cpu().numpy(), end='\n\n')
        self.test_step_outputs.clear()  # free memory

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
