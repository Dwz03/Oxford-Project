#%%

import pandas as pd
import numpy as np
from src.functions import compute_firing_rate
import os
os.chdir(r'/Users/duwenzhe/Documents/oxford_project')

# import api dependency
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
ONE.setup(base_url='https://openalyx.internationalbrainlab.org',silent=True)
one = ONE(password = "international")

region = "MOp"          # Allen CCF acronym (e.g., MOp, MOs)
need_ds = "spikes.times.npy"  # optional: only keep insertions that have this dataset

pids = one.search_insertions(atlas_acronym=region,
                            datasets=need_ds,
                            project="brainwide")   # project optional

# convert each pid to (eid, probe_label)
eid_probe = [one.pid2eid(pid) for pid in pids]
eids = [eid for eid, probe in eid_probe]

# Get the session experiment identification (EID) from PID
eids, probe= eid_probe[1]
eid = str(eids) # example eid
pid = one.eid2pid(eid)

# specify the mouse
# pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd' 

# specify experiment
# eid = one.pid2eid(pid)[0] 

# load experiment neural data
ssl = SpikeSortingLoader(eid=eid, pname = 'probe00', one=one)
spikes, clusters, channels = ssl.load_spike_sorting(spike_sorter='pykilosort') # load neural data
clusters = ssl.merge_clusters(spikes, clusters, channels) # add label information 

# load trials data to get response time
trials = one.load_object(eid, "trials")
print(trials.keys())

# Filter out NaN values, Build ONE mask for "go" trials (keep alignment across all fields)
go_mask = ~np.isnan(trials['response_times'])

# apply same mask to everything trial-aligned
response_times = trials['response_times'][go_mask] # movement is immediately after the wheel move
choices = trials['choice'][go_mask]

# Keep original trial indices too (very useful for debugging / merging later)
trial_ids = np.flatnonzero(go_mask)

# filter clusters to only those with spikes in any 200ms pre-choice windows
mask = np.zeros(len(spikes['times']), dtype=bool)
for rt in response_times:
    mask |= (spikes['times'] >= rt - 0.2) & (spikes['times'] < rt)
active_cluster_ids = np.unique(spikes['clusters'][mask]) 
print(f"Active clusters (with spikes in pre-choice windows): {len(active_cluster_ids)}")

# compute the firing_rate with 200ms within response
firing_rate = compute_firing_rate(spikes, response_times, active_cluster_ids)

# Create MultiIndex columns with (UUID, brain region)
cid_to_idx = {cid: i for i, cid in enumerate(clusters["cluster_id"])}
uuids = [clusters["uuids"][cid_to_idx[cid]] for cid in active_cluster_ids]
acronyms = [clusters["acronym"][cid_to_idx[cid]] for cid in active_cluster_ids]
neuron_cols = pd.MultiIndex.from_arrays([uuids, acronyms], names=["uuid", "acronym"])
df = pd.DataFrame(firing_rate, columns=neuron_cols, index=range(len(response_times)))

# Add aligned trial info
df.insert(0, 'trial_id', trial_ids)   # original trial index in the session
df.insert(1, 'choice', choices)       # -1 left, +1 right (0 excluded)

# Save to Excel
df.to_excel(r'data/firing_rates.xlsx')
print("Firing rates saved to firing_rates.xlsx")




#%%







