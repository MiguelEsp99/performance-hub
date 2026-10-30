"""ACWR (EWMA), semáforos de readiness y clustering de perfiles."""
from __future__ import annotations
from itertools import permutations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

LIGHT_RANK = {"🟢": 0, "⚪": 0, "🟡": 1, "🔴": 2}
PROFILES = ["Dominante Aceleración/Potencia", "Capacidad Aeróbica/Volumen", "Déficit Neuromuscular/Riesgo Isquiosural"]


def add_acwr(df: pd.DataFrame, acute: int = 7, chronic: int = 28, metrics=("player_load", "hsr")) -> pd.DataFrame:
    """EWMA: alpha = 2/(N+1). ACWR = EWMA_agudo / EWMA_crónico (primeros 14 días enmascarados)."""
    df = df.sort_values(["player_id", "date"]).copy()
    for m in metrics:
        g = df.groupby("player_id")[m]
        a = g.transform(lambda s: s.ewm(alpha=2 / (acute + 1), adjust=False).mean())
        c = g.transform(lambda s: s.ewm(alpha=2 / (chronic + 1), adjust=False).mean())
        df[f"acute_{m}"], df[f"chronic_{m}"], df[f"acwr_{m}"] = a, c, a / c
        df.loc[df.date < df.date.min() + pd.Timedelta(days=14), f"acwr_{m}"] = np.nan
    return df


def cmj_light(drop: float, asym: float) -> str:
    if drop < -10 or asym > 15: return "🔴"
    if drop < -5 or asym > 10: return "🟡"
    return "🟢"


def acwr_light(x: float) -> str:
    if pd.isna(x): return "⚪"
    if x > 1.5: return "🔴"
    if x > 1.3 or x < 0.8: return "🟡"
    return "🟢"


def readiness_table(df: pd.DataFrame, ts: pd.Timestamp, ids) -> pd.DataFrame:
    h = df[(df.date <= ts) & df.player_id.isin(ids)]
    t = h.dropna(subset=["cmj"]).sort_values("date").groupby("player_id").tail(1)
    g = t.merge(h[h.date == ts][["player_id", "acwr_player_load"]].rename(columns={"acwr_player_load": "acwr"}),
                on="player_id", how="left", suffixes=("_t", ""))
    g["cmj_drop"] = (g.cmj / g.base_cmj - 1) * 100
    g["asym"] = g.braking_asym.abs()
    g["sem_cmj"] = [cmj_light(a, b) for a, b in zip(g.cmj_drop, g.asym)]
    g["sem_acwr"] = g.acwr.map(acwr_light)
    g["global"] = [max(a, b, key=LIGHT_RANK.get) for a, b in zip(g.sem_cmj, g.sem_acwr)]
    g["_r"] = g["global"].map(LIGHT_RANK)
    g = g.sort_values(["_r", "name"], ascending=[False, True])
    return g.rename(columns={"name": "Jugador", "position": "Pos", "status": "Estado", "cmj": "CMJ (cm)",
                             "cmj_drop": "Δ CMJ vs base (%)", "asym": "Asim. frenado (%)", "acwr": "ACWR",
                             "sem_cmj": "Semáforo CMJ", "sem_acwr": "Semáforo ACWR", "global": "Global"})


def build_profiles(df: pd.DataFrame, ref: pd.Timestamp, k: int = 3) -> pd.DataFrame:
    """Agrupa por perfil funcional (KMeans sobre fuerza, velocidad, volumen) y etiqueta por centroides."""
    w = df[(df.date <= ref) & (df.date > ref - pd.Timedelta(days=28)) & (df.total_distance > 0)]
    q = lambda s: s.quantile(.75)
    p = w.groupby(["player_id", "name", "position", "status"]).agg(
        vmax=("max_speed", "max"), td=("total_distance", q), hsr=("hsr", q), accdec=("accdec", q),
        cmj=("cmj", "mean"), nord=("nord_nkg", "mean")).reset_index()
    cols = ["vmax", "cmj", "nord", "td", "hsr", "accdec"]
    z = StandardScaler().fit_transform(p[cols])
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(z)
    zc = pd.DataFrame(z, columns=cols).groupby(lab).mean()
    score = np.array([[r.vmax + r.cmj + r.accdec, r.td + r.hsr, -r.nord] for r in zc.itertuples()])
    best = max(permutations(range(k)), key=lambda perm: sum(score[c, perm[c]] for c in range(k)))
    p["perfil"] = [PROFILES[best[c]] for c in lab]
    return p
