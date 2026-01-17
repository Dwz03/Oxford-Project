# get a list of names in my region groups
# find the minimum amount of neurons in the acronym
# take the value as n, and random select the n amount of neurons in each group
# loop over the cv in three region
# compute the accuracy and make it a dataframe
# export it to excel and make a table

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
# keep_acronyms = {"MOp5", "MOp6a", "MOp6b"} # motor areas
# neuron_cols_keep = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in keep_acronyms)]

# extract X and y
meta_cols_not = [c for c in df.columns if c[0] not in meta_lvl0]
X = df[meta_cols_not].to_numpy(dtype=float)
y = (df["choice"].to_numpy() == 1).astype(int)

y = y.ravel()
n = min(np.bincount(y)) #count the number of 0 and 1 in the whole dataset, take min
# we take all trials and the area should share all trials, so we don't have to do it into different areas

# Balance the classes in the dataset, based on the lowest class count across all areas
rng = np.random.default_rng(seed=1)
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

all_results = []

for area_name, acronyms in area_subdivisions.items():
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
   

    clf = Pipeline([
        ("scaler", StandardScaler()), # normalize the train without data leakage, apply to test to make them compar
        ("logreg", LogisticRegression(max_iter=10000, penalty="l2", C=1.0, solver="lbfgs"))])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_keep, y), start=1):
        X_train, X_test = X_keep[train_idx], X_keep[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        all_results.append({
            "region": area_name,
            "fold": fold,
            "accuracy": acc})
        
    
df_cv = pd.DataFrame(all_results)
df_cv.to_excel("data/cv_accuracy_all.xlsx", index=False)

print("Saved: data/cv_accuracy_all.xlsx")

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


# hyperthalamus has lowest amount of data, so the min data is very small and could not generalize to a model
# therefore, we should only use motor and thalamus data and to compare them






    