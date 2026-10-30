"""Generador de datos sintéticos: GPS (tipo Catapult) + VALD (ForceDecks / NordBord)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MD_ORDER = ["MD-4", "MD-3", "MD-2", "MD-1", "MD", "MD+1"]
CYCLE = ["MD", "MD+1", "MD+2", "MD-4", "MD-3", "MD-2", "MD-1"]
FACTOR = {"MD": 1.0, "MD+1": 0.30, "MD+2": 0.0, "MD-4": 0.85, "MD-3": 0.65, "MD-2": 0.50, "MD-1": 0.25}
HSR_KMH, SPRINT_KMH, ACC_MS2 = 19.8, 25.2, 3.0  # umbrales GPS

POS = {  # td: distancia partido (m), ratios HSR/sprint, vmax km/h, cmj cm, nord N/kg, acc: nº acc>3m/s²
    "GK": dict(td=5200, hsr=.02, spr=.002, vmax=28.5, cmj=38, nord=3.6, acc=35),
    "CB": dict(td=9800, hsr=.06, spr=.010, vmax=31.5, cmj=40, nord=4.0, acc=70),
    "FB": dict(td=10800, hsr=.09, spr=.018, vmax=33.0, cmj=41, nord=3.9, acc=85),
    "CM": dict(td=11500, hsr=.07, spr=.012, vmax=31.8, cmj=40, nord=3.8, acc=90),
    "W": dict(td=10500, hsr=.11, spr=.025, vmax=34.0, cmj=42, nord=3.7, acc=95),
    "ST": dict(td=9500, hsr=.10, spr=.022, vmax=33.5, cmj=43, nord=4.1, acc=80),
}
SQUAD = ["GK"] * 3 + ["CB"] * 4 + ["FB"] * 4 + ["CM"] * 5 + ["W"] * 5 + ["ST"] * 4
SURNAMES = ["Álvarez", "Benítez", "Cabrera", "Díaz", "Escobar", "Fuentes", "Gómez", "Herrera", "Ibarra",
            "Juárez", "Keller", "Lozano", "Medina", "Navarro", "Ortega", "Paredes", "Quiroga", "Ramírez",
            "Salinas", "Torres", "Urrutia", "Vega", "Wagner", "Ximénez", "Zamora"]
REHAB = {5: 1, 13: 2, 18: 3}          # índice jugador -> fase RTP
RISK = {6: 1, 20: -1, 22: 1}          # sanos con perfil de riesgo isquiosural (signo de asimetría)
REHAB_LOAD = {1: .25, 2: .50, 3: .80}
REC = {1: .72, 2: .86, 3: .95}        # % recuperación funcional vs baseline
ASYM_REHAB = {1: 18, 2: 11, 3: 7}
NOTES = {
    1: ["Control del dolor, isométricos y trote lineal.", "Sin molestias en reposo. Progresión de carga excéntrica.", "Dolor leve en 70% de ROM; mantener volumen."],
    2: ["Cambios de dirección controlados, buena tolerancia.", "Introducción de estímulo caótico (rondos reducidos).", "Asimetría de frenado en descenso; reforzar unilateral."],
    3: ["Integrado en parte del grupo, sin contacto.", "Sprints >90% Vmax sin dolor. Test ForceDecks estable.", "Candidato a alta médica; validar carga semanal."],
}


def generate(seed: int = 7, days: int = 56):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(today - pd.Timedelta(days=days - 1), today)
    anchor = today - pd.Timedelta(days=4)       # hoy = MD-3
    rehab_start = today - pd.Timedelta(days=13)

    plist = []
    for i, pos in enumerate(SQUAD):
        c = POS[pos]
        ph, risk = REHAB.get(i, 0), RISK.get(i, 0)
        plist.append(dict(
            player_id=i + 1, name=f"{i + 1:02d} {SURNAMES[i]}", position=pos,
            status="Saludable" if ph == 0 else f"Fase {ph}", rehab_phase=ph,
            starter=bool(i == 0 or rng.random() < .6), mass=rng.normal(77, 6),
            vol=rng.normal(1, .06), base_vmax=c["vmax"] + rng.normal(0, 1.0),
            base_cmj=c["cmj"] + rng.normal(0, 3), base_nord=c["nord"] + rng.normal(0, .35) - (.8 if risk else 0),
            asym_base=risk * 13 if risk else rng.normal(0, 3),
            asym_sign=1 if (risk or rng.random() < .5) and risk >= 0 else -1))
    players = pd.DataFrame(plist)

    rows = []
    for p in players.itertuples():
        c = POS[p.position]
        for d in dates:
            md = CYCLE[(d - anchor).days % 7]
            f, rec, asym_mu, in_rehab = FACTOR[md], 1.0, p.asym_base, False
            if p.rehab_phase and d >= rehab_start:
                in_rehab, rec = True, REC[p.rehab_phase]
                f = 0.0 if md == "MD+2" else REHAB_LOAD[p.rehab_phase] * (.8 if md == "MD" else 1)
                asym_mu = p.asym_sign * ASYM_REHAB[p.rehab_phase]
            elif md == "MD" and not p.starter:
                f = .25
            elif md == "MD+1" and not p.starter:
                f = .85
            n = lambda s: rng.normal(1, s)
            td = c["td"] * p.vol * f * n(.05)
            test = md not in ("MD", "MD+2")
            rows.append(dict(
                date=d, player_id=p.player_id, name=p.name, position=p.position, status=p.status,
                rehab_phase=p.rehab_phase, base_cmj=p.base_cmj, md=md, in_rehab=in_rehab,
                total_distance=td, planned_total_distance=c["td"] * FACTOR[md],
                player_load=td * .092 * n(.05), planned_player_load=c["td"] * .092 * FACTOR[md],
                hsr=td * c["hsr"] * n(.12), planned_hsr=c["td"] * c["hsr"] * FACTOR[md],
                sprint=td * c["spr"] * n(.2) if f >= .6 else 0.0,
                accel=c["acc"] * f * n(.1), decel=c["acc"] * 1.1 * f * n(.1),
                max_speed=p.base_vmax * (.78 + .22 * min(f, 1)) * rec * n(.01) if f > 0 else np.nan,
                cmj=p.base_cmj * rec * n(.025) * (.96 if md == "MD+1" else 1) if test else np.nan,
                braking_asym=asym_mu + rng.normal(0, 1.5) if test else np.nan,
                nord_nkg=p.base_nord * rec * n(.03) if test else np.nan))
    df = pd.DataFrame(rows)
    df["accdec"] = df.accel + df.decel
    df["nord_n"] = df.nord_nkg * df.player_id.map(players.set_index("player_id").mass)
    df.loc[df.total_distance == 0, ["sprint", "accdec"]] = 0

    log = []
    for i, ph in REHAB.items():
        pid = i + 1
        for j, d in enumerate(pd.date_range(today - pd.Timedelta(days=13), today)):
            log.append(dict(
                player_id=pid, date=d, rpe=int(np.clip(round(rng.normal({1: 3, 2: 5, 3: 6}[ph], .8)), 1, 10)),
                pain=int(np.clip(round(rng.normal({1: 5.5, 2: 3.5, 3: 2.0}[ph] - .25 * j, .6)), 0, 10)),
                note=NOTES[ph][j % 3]))
    return players, df, pd.DataFrame(log)
