#%%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.chdir(r'/Users/duwenzhe/Documents/oxford_project')

# import the file
df = pd.read_excel("data/firing_rates.xlsx", header=[0, 1], index_col=0) # we have an index_col

# Meta columns to keep
meta_lvl0 = {"choice", "trial_id", "uuid"}
meta_cols = [c for c in df.columns if c[0] in meta_lvl0]

# Only keep neurons from specific brain areas (optional)
keep_acronyms = {"MOp5", "MOp6a", "MOp6b"} # motor areas
neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

# extract X and y
X = df.loc[:, neuron_cols_keep].to_numpy(dtype=float)
y = (df["choice"].to_numpy() == 1).astype(int)

# Define the region of the brain
region_groups = {
    "motor": {"MOp5", "MOp6a", "MOp6b"},
    "thalamus": {"AD", "AMd", "AMv", "AV", "IAD", "PR"},
    "hypothalamus": {"ZI"}
}

# create a cross-validation pipeline
clf = Pipeline([
    ("scaler", StandardScaler()), # normalize the train without data leakage, apply to test to make them compar
    ("logreg", LogisticRegression(
        max_iter=10000, penalty="l2", C=1.0, solver="lbfgs"
    ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# create a list that contain information for three area
all_results = []

# create the X for motor region
keep_acronyms = region_groups["motor"]
neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

# extract X
X = df.loc[:, neuron_cols_keep].to_numpy(dtype=float)

# run the cv over motor area
fold_acc = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_acc.append(acc)

    print(f"Fold {fold}: acc = {acc:.3f}")

    all_results.append({
        "region": "motor",
        "fold": fold,
        "accuracy": acc
    })

print("Motor mean:", np.mean(fold_acc))
print("Motor std :", np.std(fold_acc, ddof=1))

# now do the same for thalamus area
keep_acronyms = region_groups["thalamus"]
neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

X = df.loc[:, neuron_cols_keep].to_numpy(dtype=float)

fold_acc = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_acc.append(acc)

    print(f"Fold {fold}: acc = {acc:.3f}")

    all_results.append({
        "region": "thalamus",
        "fold": fold,
        "accuracy": acc
    })

print("Thalamus mean:", np.mean(fold_acc))
print("Thalamus std :", np.std(fold_acc, ddof=1))

# now same for hypothalamus
keep_acronyms = region_groups["hypothalamus"]

neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

X = df.loc[:, neuron_cols_keep].to_numpy(dtype=float)

fold_acc = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fold_acc.append(acc)

    print(f"Fold {fold}: acc = {acc:.3f}")

    all_results.append({
        "region": "hypothalamus",
        "fold": fold,
        "accuracy": acc
    })

print("hypothalamus mean:", np.mean(fold_acc))
print("hypothalamus std :", np.std(fold_acc, ddof=1))

# create a dataframe for all brain region result
df_cv = pd.DataFrame(all_results)
df_cv.to_excel("data/cv_accuracy_all.xlsx", index=False)

print("Saved: data/cv_accuracy_all.xlsx")

# import the accuracy
df_acc = pd.read_excel("data/cv_accuracy_all.xlsx")

# create a table for the summary
summary = (
    df_acc.groupby("region")["accuracy"]
      .agg(["mean", "std"])
      .reset_index()
)

print(summary)

# create a plot
regions = summary["region"].tolist()
means = summary["mean"].values
stds = summary["std"].values

x = np.arange(len(regions))

plt.figure(figsize=(6, 4))

# bar + error bars
plt.bar(x, means, yerr=stds, capsize=6, alpha=0.7)

# scatter individual folds
for i, region in enumerate(regions):
    vals = df_acc.loc[df_acc["region"] == region, "accuracy"]
    jitter = np.random.normal(0, 0.04, size=len(vals))  # small x jitter
    plt.scatter(np.full_like(vals, x[i]) + jitter, vals,
                color="black", s=30, zorder=10)

plt.xticks(x, regions)
plt.ylabel("Decoding accuracy")
plt.title("Choice decoding by brain region")

plt.ylim(0.65, 0.9)   #  
plt.tight_layout()
plt.show()

# %%

# in the context of controlling variable, we should control the amount of data to make comparason fair
# use same amout of neurons, use amount of 0's or 1's that is smaller. My data should consist the exact same amount of
# trials in each class in order to balance within group and class.


