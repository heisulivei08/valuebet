"""Ottelusimulaatio: 90 minuuttia kerrallaan, tilanne vaikuttaa tahtiin, punaiset kortit mukana."""
import numpy as np
import pandas as pd

PERIODS = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90"]


def simulate(lam_h, lam_a, ref_c=1.0, n=20000, seed=7):
    """lam_h / lam_a = odotetut maalit. ref_c = tuomarin korttikerroin (1.0 = keskiarvo)."""
    rng = np.random.default_rng(seed)
    shape = 1 + 0.25 * (np.arange(90) / 90 - 0.5)  # maaleja hieman enemmän loppua kohti
    shape /= shape.mean()
    gh, ga = np.zeros(n, int), np.zeros(n, int)
    rh, ra = np.zeros(n, bool), np.zeros(n, bool)
    first = np.zeros(n, int)
    p_red = 0.07 * ref_c / 90  # oletus: ~7 % todennäköisyys punaiselle per joukkue per ottelu
    bucket, lead = np.zeros((2, 6)), []
    for t in range(90):
        d = gh - ga
        mh = np.where(d < 0, 1.12, np.where(d > 0, 0.94, 1.0)) * np.where(rh, 0.7, 1) * np.where(ra, 1.25, 1)
        ma = np.where(d > 0, 1.12, np.where(d < 0, 0.94, 1.0)) * np.where(ra, 0.7, 1) * np.where(rh, 1.25, 1)
        sh = rng.random(n) < lam_h / 90 * shape[t] * mh
        sa = rng.random(n) < lam_a / 90 * shape[t] * ma
        first = np.where((first == 0) & sh, 1, np.where((first == 0) & sa, 2, first))
        gh += sh
        ga += sa
        bucket[0, t // 15] += sh.sum()
        bucket[1, t // 15] += sa.sum()
        rh |= rng.random(n) < p_red
        ra |= rng.random(n) < p_red
        if t % 15 == 14:
            lead.append(((gh > ga).mean(), (gh == ga).mean(), (gh < ga).mean()))
    tot = gh + ga
    return dict(
        p=((gh > ga).mean(), (gh == ga).mean(), (gh < ga).mean()),
        goals=(gh.mean(), ga.mean()),
        ou={f"Yli {l}": (tot > l).mean() for l in (1.5, 2.5, 3.5)},
        btts=((gh > 0) & (ga > 0)).mean(),
        first=((first == 1).mean(), (first == 2).mean(), (first == 0).mean()),
        red=(rh | ra).mean(),
        scores=pd.Series([f"{a}-{b}" for a, b in zip(gh, ga)]).value_counts(normalize=True).head(8),
        timing=pd.DataFrame(bucket.T / n, index=PERIODS, columns=["Koti", "Vieras"]),
        lead=pd.DataFrame(lead, index=PERIODS[:0] or [15, 30, 45, 60, 75, 90],
                          columns=["Koti johtaa", "Tasan", "Vieras johtaa"]),
    )
