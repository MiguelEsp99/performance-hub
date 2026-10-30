"""Dashboard Streamlit: GPS + VALD para microciclo competitivo y Return-To-Play.  Ejecutar: streamlit run app.py"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import analytics as an
import charts as ch
from data_gen import MD_ORDER, generate

st.set_page_config(page_title="Performance Hub", page_icon="⚽", layout="wide")
st.markdown("""<style>
.stApp{background:#0b1220} [data-testid=stSidebar]{background:#0f172a}
[data-testid=stMetric]{background:#111c33;border:1px solid #1e3a5f;border-left:4px solid #22d3ee;padding:12px;border-radius:10px}
h1,h2,h3{color:#e2e8f0} </style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    players, df, log = generate()
    return players, an.add_acwr(df), log


players, df, rlog = load()
st.title("⚽ Performance Hub · GPS + Fuerza/Salto")
st.caption("Datos sintéticos (mock). Umbrales: HSR > 19.8 km/h · Sprint > 25.2 km/h · Acc/Dec > 3 m/s²")

# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("Filtros globales")
ts = pd.Timestamp(sb.date_input("Fecha de análisis", df.date.max().date(), df.date.min().date(), df.date.max().date()))
mds = sb.multiselect("Microciclo (MD) – tendencia", MD_ORDER, MD_ORDER)
pos = sb.multiselect("Posición", sorted(players.position.unique()), sorted(players.position.unique()))
sts = sb.multiselect("Estado", ["Saludable", "Fase 1", "Fase 2", "Fase 3"], ["Saludable", "Fase 1", "Fase 2", "Fase 3"])
P = players[players.position.isin(pos) & players.status.isin(sts)]
who = sb.selectbox("Jugador", ["Todos"] + P.name.tolist())
if who != "Todos":
    P = P[P.name == who]
ids = P.player_id.tolist()
if not ids:
    st.warning("Sin jugadores para los filtros seleccionados."); st.stop()

t1, t2, t3, t4 = st.tabs(["📊 Squad & Readiness", "🧬 Perfilado & Clustering", "📈 Carga & ACWR", "🩺 Clínico & RTP"])

# ---------------- Tab 1 ----------------
with t1:
    day = df[(df.date == ts) & df.player_id.isin(ids)]
    grid = an.readiness_table(df, ts, ids)
    alerts = int((grid["Global"] == "🔴").sum())
    c = st.columns(4)
    c[0].metric("Disponibles", int((P.status == "Saludable").sum()))
    c[1].metric("En readaptación", int((P.status != "Saludable").sum()))
    c[2].metric("PlayerLoad total (día)", f"{day.player_load.sum():,.0f}")
    c[3].metric("Alertas activas (rojo)", alerts)

    st.subheader("Readiness Grid · CMJ (VALD) × ACWR (GPS)")
    paint = lambda v: {"🟢": "background-color:#065f46", "🟡": "background-color:#92400e", "🔴": "background-color:#991b1b"}.get(v, "")
    show = grid[["Jugador", "Pos", "Estado", "CMJ (cm)", "Δ CMJ vs base (%)", "Asim. frenado (%)", "ACWR",
                 "Semáforo CMJ", "Semáforo ACWR", "Global"]]
    st.dataframe(show.style.map(paint, subset=["Semáforo CMJ", "Semáforo ACWR", "Global"])
                 .format({"CMJ (cm)": "{:.1f}", "Δ CMJ vs base (%)": "{:+.1f}", "Asim. frenado (%)": "{:.1f}", "ACWR": "{:.2f}"}),
                 hide_index=True, use_container_width=True, height=420)
    st.caption("CMJ: 🔴 caída >10% o asimetría >15% · 🟡 >5% / >10%. ACWR: 🟢 0.8–1.3 · 🟡 <0.8 o 1.3–1.5 · 🔴 >1.5.")

    mets = {"PlayerLoad": ("player_load", "planned_player_load"), "Distancia Total (m)": ("total_distance", "planned_total_distance"),
            "HSR (m)": ("hsr", "planned_hsr")}
    m = st.radio("Métrica del microciclo", list(mets), horizontal=True)
    w = df[(df.date <= ts) & (df.date > ts - pd.Timedelta(days=28)) & df.player_id.isin(ids) & (df.status == "Saludable") & df.md.isin(mds)]
    agg = w.groupby("md")[list(mets[m])].mean().reindex([x for x in MD_ORDER if x in mds]).dropna()
    agg.columns = ["real", "planned"]
    st.plotly_chart(ch.microcycle_chart(agg, m), use_container_width=True)

# ---------------- Tab 2 ----------------
with t2:
    prof = an.build_profiles(df, ts)
    pv = prof[prof.player_id.isin(ids)]
    st.plotly_chart(ch.scatter_quadrants(pv), use_container_width=True)
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Perfiles funcionales")
        st.dataframe(pv[["name", "position", "perfil", "vmax", "nord", "cmj"]].rename(columns={
            "name": "Jugador", "position": "Pos", "perfil": "Perfil", "vmax": "Vmax", "nord": "N/kg", "cmj": "CMJ"})
            .sort_values("Perfil").style.format({"Vmax": "{:.1f}", "N/kg": "{:.2f}", "CMJ": "{:.1f}"}),
            hide_index=True, use_container_width=True, height=440)
    with b:
        sel = st.selectbox("Radar individual", pv.name.tolist(), key="radar")
        cols = ["vmax", "cmj", "nord", "td", "hsr", "accdec"]
        cats = ["Vmax", "CMJ", "NordBord", "Volumen (TD)", "HSR", "Acc+Dec"]
        norm = prof[cols] / prof[cols].max() * 100
        r = prof[prof.name == sel].iloc[0]
        st.plotly_chart(ch.radar(cats, norm.loc[r.name].tolist(), norm[prof.position == r.position].mean().tolist(), sel, r.position),
                        use_container_width=True)

# ---------------- Tab 3 ----------------
with t3:
    names = P.name.tolist()
    c1, c2 = st.columns(2)
    pl = c1.selectbox("Jugador", names, key="acwr_p")
    mt = c2.radio("Métrica", ["player_load", "hsr"], format_func=lambda x: {"player_load": "PlayerLoad", "hsr": "HSR (m)"}[x], horizontal=True)
    d = df[(df.name == pl) & (df.date <= ts) & (df.date > ts - pd.Timedelta(days=28))]
    last = d.iloc[-1]
    k = st.columns(3)
    k[0].metric("Carga aguda (EWMA 7d)", f"{last[f'acute_{mt}']:.0f}")
    k[1].metric("Carga crónica (EWMA 28d)", f"{last[f'chronic_{mt}']:.0f}")
    k[2].metric("ACWR", f"{last[f'acwr_{mt}']:.2f}" if pd.notna(last[f'acwr_{mt}']) else "–")
    st.plotly_chart(ch.acwr_chart(d, mt, "PlayerLoad" if mt == "player_load" else "HSR (m)", f"acwr_{mt}"), use_container_width=True)

# ---------------- Tab 4 ----------------
with t4:
    rh = P[P.status != "Saludable"]
    if rh.empty:
        st.info("No hay jugadores en readaptación con los filtros actuales.")
    else:
        pn = st.selectbox("Jugador en RTP", rh.name.tolist())
        pr = rh[rh.name == pn].iloc[0]; ph = int(pr.rehab_phase)
        labels = ["Fase 1 · Control", "Fase 2 · Control-Caos", "Fase 3 · Retorno al grupo"]
        for i, col in enumerate(st.columns(3), 1):
            col.markdown(f"<div style='padding:10px;border-radius:8px;text-align:center;border:1px solid "
                         f"{'#22d3ee' if i == ph else '#1e293b'};background:{'#0e3a4a' if i == ph else '#111c33'}'>"
                         f"{'✅ ' if i < ph else '▶ ' if i == ph else ''}{labels[i - 1]}</div>", unsafe_allow_html=True)
        h = df[(df.name == pn) & (df.date <= ts)]
        pre, ses = h[~h.in_rehab & (h.total_distance > 0)], h[h.in_rehab & h.cmj.notna() & (h.total_distance > 0)]
        if ses.empty:
            st.warning("Sin sesión de readaptación con test en la fecha seleccionada.")
        else:
            s = ses.iloc[-1]
            pct = {"Distancia": s.total_distance / pre.total_distance.mean(), "% Vmax": s.max_speed / pre.max_speed.max(),
                   "HSR": s.hsr / pre.hsr.mean(), "CMJ": s.cmj / pre.cmj.mean(), "NordBord N/kg": s.nord_nkg / pre.nord_nkg.mean()}
            vals = [v * 100 for v in pct.values()]
            a, b = st.columns([3, 1])
            a.plotly_chart(ch.benchmark_chart(list(pct), vals), use_container_width=True)
            asym = abs(s.braking_asym)
            b.metric("Asimetría frenado (ForceDecks)", f"{asym:.1f}%", f"L {50 + asym / 2:.0f}% / R {50 - asym / 2:.0f}%", delta_color="off")
            ok = min(vals[3:]) >= 90 and asym < 10
            b.success("Criterios de fuerza/salto cumplidos") if ok else b.warning("Criterios RTP pendientes (≥90% y asim. <10%)")
        lg = rlog[rlog.player_id == pr.player_id]
        c1, c2 = st.columns(2)
        c1.plotly_chart(ch.pain_chart(lg), use_container_width=True)
        with c2:
            st.subheader("Notas clínicas")
            store = st.session_state.setdefault("notes", {})
            mine = store.setdefault(pr.player_id, lg[["date", "note"]].tail(5).assign(date=lambda x: x.date.dt.strftime("%d/%m")).values.tolist())
            new = st.text_area("Nueva nota", key=f"n{pr.player_id}", height=80)
            if st.button("Guardar nota") and new.strip():
                mine.append([ts.strftime("%d/%m"), new.strip()]); st.rerun()
            for dte, txt in reversed(mine):
                st.markdown(f"**{dte}** · {txt}")
