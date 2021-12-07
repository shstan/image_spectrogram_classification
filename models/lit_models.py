import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as T
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchvision import models
from transforms.transform import IdentityTransform

def stat_format(info_dict):
    join_lst = []
    for key in info_dict:
        join_lst.append(f"{key}={info_dict[key]:.5f}")
    return '[' + ', '.join(join_lst) + ']'

def stat_best_format(info_dict):
    join_lst = []
    for key in info_dict:
        join_lst.append(f"{key}={info_dict[key][0]:.5f}(epoch:{info_dict[key][1]})")
    return '[' + ', '.join(join_lst) + ']'



class CNN_classifier(pl.LightningModule):
    def __init__(self, encoder, input_dim=512, hidden_dim=512,
                 num_logits=1, dropout_p=0.1, lr=1e-3, no_fc=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_logits = num_logits
        self.encoder = encoder
        self.no_fc = no_fc
        if not no_fc:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True), # hidden layer
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, num_logits, bias=False)
            )
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.last_epoch = 0
        self.best_val_stats = {
                            "epoch_avg_val_loss":(float('inf'), -1),
                            "VAL_ROC_AUC": (float('-inf'), -1),
                            "VAL_F1": (float('-inf'), -1),
                            "VAL_Accuracy": (float('-inf'), -1)
                        }
            
    def get_encoder(self):
        return self.encoder
    
    def forward(self, x):
        x = self.encoder(x) # NxC
        if not self.no_fc:
            x = self.fc(x) # NxC
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.flatten(), y.float())
        train_loss_log={"train_loss_batch":loss.item()}
        self.log('train_loss', train_loss_log, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self,outputs):
        # input(outputs)
        avg_loss = np.array([x['loss'].item() for x in outputs]).mean()
        self.log('loss',
                 {
                     "epoch_avg_train_loss":avg_loss
                 },
                 on_step=False, on_epoch=True)
        print(stat_format({
                     "epoch_avg_train_loss":avg_loss
                 }))
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.flatten(), y.float())
        train_loss_log={"val_loss_batch":loss.item()}
        batch_dict = {
            'loss':loss,
            'pred_labels':logits.sigmoid().detach().flatten().cpu().numpy(),
            'true_labels': y.detach().flatten().cpu().numpy()
        }
        self.log('val_loss', train_loss_log, on_step=True, on_epoch=True)
        return batch_dict

    def validation_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        pred_labels = np.concatenate([x['pred_labels'] for x in outputs], axis=0)
        pred_rounded = np.array([1 if x >= 0.5 else 0 for x in pred_labels])
        true_labels = np.concatenate([x['true_labels'] for x in outputs], axis=0)
        auc = roc_auc_score(y_score=pred_labels, y_true=true_labels)
        f1 = f1_score(y_true=true_labels, y_pred=pred_rounded)
        acc = accuracy_score(y_true=true_labels, y_pred=pred_rounded)
        # calculating average loss  
        avg_loss = np.array([x['loss'].item() for x in outputs]).mean()
        val_log_info = {
            "epoch_avg_val_loss":avg_loss,
            "VAL_ROC_AUC": auc,
            "VAL_F1": f1,
            "VAL_Accuracy": acc
        }
        self.log('loss',val_log_info,on_step=False, on_epoch=True)
        print(stat_format(val_log_info))
        for key in val_log_info:
            if key=="epoch_avg_val_loss":
                if self.best_val_stats[key][0] > val_log_info[key]:
                    self.best_val_stats[key] = (val_log_info[key], self.last_epoch)
            else:
                if self.best_val_stats[key][0] < val_log_info[key]:
                    self.best_val_stats[key] = (val_log_info[key], self.last_epoch)
        print("best_stat:", stat_best_format(self.best_val_stats)) 
        self.last_epoch += 1


class SimSiam(pl.LightningModule):
    def __init__(self,
                 encoder,
                 transforms,
                 predictor=None,
                 dim=512, pred_dim=256, lr=1e-3):
        super(SimSiam, self).__init__()
        self.dim = dim
        self.encoder = encoder
        if predictor is None:
            self.predictor = nn.Sequential(
                nn.Linear(dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True), # hidden layer
                nn.Linear(pred_dim, dim)
            )
        else:
            self.predictor = predictor
        
        self.criterion = nn.CosineSimilarity(dim=1)
        self.transforms = transforms
        self.lr = lr
    
    def get_encoder(self):
        return self.encoder
    
    def forward(self, x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        X, y = batch
        p1, p2, z1, z2 = self(X)
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        train_loss_log={"train_loss_batch":loss.item()}
        batch_dict = {'log':train_loss_log, 'loss':loss}
        self.log('loss', train_loss_log, on_step=True, on_epoch=True)
        return batch_dict

    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('loss',{"epoch_avg_train_loss":avg_loss}, on_step=False, on_epoch=True)