import numpy as np
from sklearn.metrics import adjusted_rand_score

def get_bootstrap_ari(a_set, b_set, bn):
    a_ns = len(a_set)
    b_ns = len(b_set)

    scores = np.ones((a_ns, b_ns)) * np.nan
    bs_sample = np.zeros(bn)
    for i in range(bn):
        ai = np.random.randint(0, a_ns)
        bi = np.random.randint(0, b_ns)

        if np.isnan(scores[ai, bi]):
            scores[ai, bi] = adjusted_rand_score(a_set[ai], b_set[bi])

        bs_sample[i] = scores[ai, bi]

    return bs_sample.mean(), bs_sample.std()
