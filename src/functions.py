import numpy as np

def compute_firing_rate(spikes, response_times, active_cluster_ids):
        
    win = 0.2 # time window

    # sort response times
    rt_order = np.argsort(response_times)
    rt = response_times[rt_order]

    # sort spikes by time
    spk_order = np.argsort(spikes['times'])
    st = spikes['times'][spk_order]
    sc = spikes['clusters'][spk_order]

    active_cluster_ids = np.sort(np.asarray(active_cluster_ids))

    # next response time for each spike (can be len(rt) if spike is after last rt)
    trial_idx = np.searchsorted(rt, st, side="left")

    # IMPORTANT: drop spikes after the last response time BEFORE indexing rt[trial_idx]
    valid_trial = trial_idx < len(rt)
    trial_idx_v = trial_idx[valid_trial]
    st_v = st[valid_trial]
    sc_v = sc[valid_trial]

    # now safe to index rt[trial_idx_v]
    valid_time = (st_v >= (rt[trial_idx_v] - 2)) & (st_v < rt[trial_idx_v] - 1.8) # change the window

    trial_idx_v = trial_idx_v[valid_time]
    cids = sc_v[valid_time]

    # map cluster IDs -> columns
    cols = np.searchsorted(active_cluster_ids, cids)
    ok = (cols < len(active_cluster_ids)) & (active_cluster_ids[cols] == cids)

    trial_idx_v = trial_idx_v[ok]
    cols = cols[ok]

    # accumulate spike counts
    counts = np.zeros((len(rt), len(active_cluster_ids)), dtype=np.int32)
    np.add.at(counts, (trial_idx_v, cols), 1)

    firing_rates_sorted = counts / win

    # unsort to original response_times order
    firing_rates = np.empty_like(firing_rates_sorted, dtype=np.float32)
    firing_rates[rt_order, :] = firing_rates_sorted

    return firing_rates

# the time window [firstmovement_time - 0.2, firstmovement_time] create a better prediction accuracy
# than the time window [firstmovement_time - 0.4, firstmovement_time - 0.2]