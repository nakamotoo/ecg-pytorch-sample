import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import ecg

print("\nNEW EXPERIMENT!!", sys.argv[1], sys.argv[2], sys.argv[3])

filelist_name = "FileList_{}.csv".format(sys.argv[3])
data_folder="./data"

mean, std = ecg.utils.get_mean_and_std(ecg.datasets.ECG(split="train", filelist_name=filelist_name, data_folder=data_folder), num_workers=8)

# Model_2D. Model_1DRes, Model_WaveNet, Model_TFEncoder
modelname = sys.argv[1]
nhead = int(sys.argv[4])
nlayers = int(sys.argv[5])

output = os.path.join("output_debug", sys.argv[3], "{}_{}_h{}_l{}".format(modelname, sys.argv[2], nhead, nlayers))
os.makedirs(output, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ecg.models.__dict__[modelname](nhead=nhead, nlayers=nlayers)
net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.to(device)

train_dataset = ecg.datasets.ECG(split="train", mean=mean, std=std, filelist_name=filelist_name , data_folder=data_folder)
val_dataset = ecg.datasets.ECG(split="valid", mean=mean, std=std, filelist_name=filelist_name , data_folder=data_folder)

dataloader_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size= 8,
    shuffle=True,
    num_workers = 8
)

dataloader_valid = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True,
    num_workers = 8
)

pos_weight = (len(train_dataset) - sum(train_dataset.targets)) / sum(train_dataset.targets)

n_epochs = 200
lr = 1e-5

if sys.argv[2] == "adam":
    optimizer =optim.Adam(net.parameters(), lr=lr)
elif sys.argv[2] == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

loss_function = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
early_stopping = ecg.utils.EarlyStopping(patience=2, verbose=True, path=os.path.join(output, "best.pt"))

losses_train_all = []
losses_valid_all = []

with open(os.path.join(output, "log.csv"), "w") as f:
    for epoch in range(n_epochs):
        total_loss_train = 0
        total_loss_val = 0

        t_true_train = []
        t_pred_train = []

        t_true_valid = []
        t_pred_valid = []

        net.train()

        n_train = 0
        acc_train = 0

        for x1, t in tqdm(dataloader_train):
            n_train += t.size()[0]
            t_true_train.extend(t.tolist())

            optimizer.zero_grad()

            x1= x1.to(device)
            t = t.reshape(-1, 1).type(torch.DoubleTensor).to(device)

            y, _ = net(x1)
            loss = loss_function(y, t)
            total_loss_train += loss.sum().item()
            loss.mean().backward()
            optimizer.step()

            pred_prob = torch.sigmoid(y)
            t_pred_train.extend(pred_prob.reshape(-1).tolist())

            pred = np.where(pred_prob.to('cpu').detach().numpy().copy() < 0.5, 0, 1)
            acc_train += (pred == t.to('cpu').detach().numpy().astype("float")).sum().item()

        net.eval()
        n_val = 0
        acc_val = 0

        for x1, t in tqdm(dataloader_valid):
            n_val += t.size()[0]
            t_true_valid.extend(t.tolist())

            x1= x1.to(device)
            t = t.reshape(-1, 1).type(torch.DoubleTensor).to(device)

            y, _ = net(x1)

            loss = loss_function(y, t)
            total_loss_val += loss.sum().item()

            pred_prob = torch.sigmoid(y)
            t_pred_valid.extend(pred_prob.reshape(-1).tolist())

            pred = np.where(pred_prob.to('cpu').detach().numpy()< 0.5, 0, 1)
            acc_val += (pred == t.to('cpu').detach().numpy().astype("float")).sum().item()

        losses_train_all.append(total_loss_train / n_train)
        losses_valid_all.append(total_loss_val / n_val)
        scheduler.step(total_loss_val / n_val)
        early_stopping(total_loss_val / n_val, net)

        try:
            auc_train = roc_auc_score(t_true_train, t_pred_train)
        except:
            auc_train = 0

        try:
            auc_valid = roc_auc_score(t_true_valid, t_pred_valid)
        except:
            auc_valid = 0


        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}, AUC: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}, AUC: {:.3f}]'.format(
            epoch,
            total_loss_train / n_train,
            acc_train/n_train,
            auc_train,
            total_loss_val / n_val,
            acc_val/n_val,
            auc_valid
        ))

        torch.save(net.state_dict(), os.path.join(output, "latest.pt"))

        f.write("{},{},{},{},{}\n".format(
            epoch,
            total_loss_train / n_train,
            auc_train,
            total_loss_val / n_val,
            auc_valid))
        f.flush()

        if early_stopping.early_stop:
            f.close()
            print("Early Stopped.")
            break
    f.close()

test_dataset = ecg.datasets.ECG(split="test", mean=mean, std=std, filelist_name=filelist_name , data_folder=data_folder)
dataloader_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers = 16
)

checkpoint = torch.load(os.path.join(output, "best.pt"))
net.load_state_dict(checkpoint)
net.eval()

t_true_test = []
t_pred_test= []
n_test = 0
acc_test = 0
total_loss_test = 0

for x1, t in tqdm(dataloader_test):
    n_test += t.size()[0]
    t_true_test.extend(t.tolist())

    x1= x1.to(device)
    t = t.reshape(-1, 1).type(torch.DoubleTensor).to(device)

    y, _ = net(x1)

    loss = loss_function(y, t)
    total_loss_test += loss.sum().item()

    pred_prob = torch.sigmoid(y)
    t_pred_test.extend(pred_prob.reshape(-1).tolist())

    pred = np.where(pred_prob.to('cpu').detach().numpy()< 0.5, 0, 1)
    acc_test += (pred == t.to('cpu').detach().numpy().astype("float")).sum().item()


try:
    auc_test = roc_auc_score(t_true_test, t_pred_test)
except:
    auc_test = 0


print('Test [Loss: {:.3f}, Accuracy: {:.3f}, AUC: {:.3f}]'.format(
        total_loss_test / n_test,
        acc_test/n_test,
        auc_test
    ))


fig = plt.figure()

plt.plot(losses_train_all, linewidth=3, label="train")
plt.plot(losses_valid_all, linewidth=3, label="validation")
plt.title("Learning Curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output, "fig1.png"))
plt.clf()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(t_true_test, t_pred_test)
fig = plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Test AUC : {}".format(str(roc_auc_score(t_true_test, t_pred_test))))
plt.grid()
plt.savefig(os.path.join(output, "fig2.png"))

np.save(os.path.join(output,"train_loss"), losses_train_all)
np.save(os.path.join(output,"val_loss"), losses_valid_all)
