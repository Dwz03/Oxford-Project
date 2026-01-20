#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

os.chdir(r'/Users/duwenzhe/Documents/oxford_project')

batch_size = 64
epochs = 500 # number of training epochs, you can change this and see how it affects performance.
lr = 1e-3 # learning rate, change and see how it affects performance.
weight_decay = 1e-4# set to e.g., 1e-4 for L2 regularization
hidden_dim = 64
dropout = 0.2  # set e.g. 0.2 if you want regularisation
rng = np.random.default_rng(seed=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import the file
df = pd.read_excel("data/firing_rates.xlsx", header=[0, 1], index_col=0) # we have an index_col

# Meta columns to keep
meta_lvl0 = {"choice", "trial_id", "uuid"}
meta_cols = [c for c in df.columns if c[0] in meta_lvl0]

# extract X and y
meta_cols_not = [c for c in df.columns if c[0] not in meta_lvl0]
X = df[meta_cols_not].to_numpy(dtype=float)
y = (df["choice"].to_numpy() == 1).astype(int)

y = y.ravel()
n = min(np.bincount(y))

ng = np.random.default_rng(seed=1)
idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]

keep = np.concatenate([
    rng.choice(idx0, n, replace=False),
    rng.choice(idx1, n, replace=False),
])

X, y = X[keep], y[keep]

neuron_per_area = []

area_subdivisions = {
    "motor": {"MOp5", "MOp6a", "MOp6b"},
    "thalamus": {"AD", "AMd", "AMv", "AV", "IAD", "PR"},
    # "hypothalamus": {"ZI"}
}

for area_name, acronyms in area_subdivisions.items():
    neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in acronyms)]
    m = len(neuron_cols_keep)
    neuron_per_area.append(m)
    print(area_name, acronyms)

min_neu_per_area = min(neuron_per_area)

# build a pipeline that takes all the data and split it and then use feedforward network to train it

area_fold_acc = []
for area_name, acronyms in area_subdivisions.items():
    fold_acc = [] 
    neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in acronyms)]
    neuron_cols_keep = rng.choice(neuron_cols_keep,min_neu_per_area,replace = False)
    X_keep = df[neuron_cols_keep].to_numpy()
    rng = np.random.default_rng(seed=1)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    keep = np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False),])
    
    X_keep,y = X_keep[keep],y[keep]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def standardize_with_train_stats(X_train: np.ndarray, X_test: np.ndarray):
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0, ddof=0)
            std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero
            return (X_train - mean) / std, (X_test - mean) / std
    
    # play with more nodes and more layers, and activation functions. 
    # play withn different windows for your data to see if model predictions start to differ. 
    
    class SimpleMLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 240, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 classes (0/1)
            )

        def forward(self, x):
            return self.net(x)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_keep, y), start=1):
        X_train, X_test = X_keep[train_idx], X_keep[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fold-wise standardize
        X_train, X_test = standardize_with_train_stats(X_train, X_test)

        # tensors
        Xtr = torch.tensor(X_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.long) 
        Xte = torch.tensor(X_test, dtype=torch.float32)
        yte = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        # tensordataset prepares our dimension be formatted in ways dataloader could work
        # dataloader help processing 

        # model
        model = SimpleMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)

        # Loss funcrion
        criterion = nn.CrossEntropyLoss() # the loss is what we optimise

        # Optimiser (Adam with weight decay for L2 regularization)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # weight decay applys some amount of depression 
        # so it don't become large. Keep weight before becoming too large

        # train
        loss_store = []
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader: # in trainloader, load in batches
                xb, yb = xb.to(device), yb.to(device) # use this data format in what device we want to sue
                optim.zero_grad(set_to_none=True) # clear gradient to 0 every epochs
                logits = model(xb) # data in that batch, model input
                loss = criterion(logits, yb) 
                # print(logits), print(yb)
                loss.backward() # start backward propagation to compute each gradient of bias and weights
                optim.step() # we update weights and bias trying to minimise the loss, using 
                # information on loss.backward()
            loss_store.append(float(loss.detach()))
        # test
        model.eval() # in testing mode
        with torch.no_grad(): # without consider grad
            logits = model(Xte.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy() # index of highest value
        acc = float((pred == y_test).mean())
        fold_acc.append(acc)

        print(f"Fold {fold}/{5} accuracy: {acc:.3f}")

    area_fold_acc.append(fold_acc)


    fold_accs = np.array(fold_acc, dtype=float)
    acc_m = float(fold_accs.mean())
    acc_s = float(fold_accs.std(ddof=1))

    print("\n5-fold cross-validation performance (test):")
    print(f"  Accuracy: {acc_m:.3f} ± {acc_s:.3f}")
    print("Fold accuracies:", np.round(fold_accs, 3))

    # diagnostic plot of loss over epochs 
    fig, ax = plt.subplots(1,1)
    ax.plot(range(len(loss_store)), loss_store)
    plt.show()

# save
df_acc = pd.DataFrame({"motor cortex": area_fold_acc[0],
                       "thalamus": area_fold_acc[1],
                    })
df_acc.to_excel('results/mlp_decoding.xlsx', index=False)
print("Saved: results/mlp_decoding.xlsx")


df_acc = pd.read_excel("results/mlp_decoding.xlsx")


# create a plot
regions = df_acc.columns
means = df_acc.mean()
stds = df_acc.std() 
sem = stds / np.sqrt(len(df_acc))

x = np.arange(len(regions))

fig,ax = plt.subplots(1,1,figsize = (4,3), dpi = 1200)

# bar + error bars
plt.bar(x, means, yerr=sem, capsize=6, alpha=0.7)

# scatter individual folds
for i, region in enumerate(regions):
    vals = df_acc[region]
    jitter = np.random.normal(0, 0.04, size=len(vals))  # small x jitter
    plt.scatter(np.full_like(vals, x[i]) + jitter, vals,
                color="black", s=30, zorder=10)

plt.xticks(x, regions)
plt.ylabel(u"Decoding accuracy \u00B1 sem")
plt.title("feedfoward neural network")
ax.spines[['right','top']].set_visible(False)
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# number of folds should be larger
# they are no overlapping means in t test statistically, it is not overlapping

# if I add more layer, the loss function would be much more smoother and fewer spikes


# to work with GNN
# find the sequence of firing rates/spikes, look at correlation of neurons across multiple trials
# adapt dataloading function, use the data in a GNN

        
    