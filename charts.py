"""Gráficos Plotly en dark mode (cian / esmeralda / ámbar)."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CYAN, EMERALD, AMBER, RED, SLATE = "#22d3ee", "#10b981", "#f59e0b", "#ef4444", "#94a3b8"
PROFILE_COLORS = {"Dominante Aceleración/Potencia": CYAN, "Capacidad Aeróbica/Volumen": EMERALD,
                  "Déficit Neuromuscular/Riesgo Isquiosural": AMBER}


def _style(fig: go.Figure, title: str | None = None, h: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", height=h,
                      title=title, margin=dict(l=10, r=10, t=50, b=10), font=dict(color="#e2e8f0"),
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="#1e293b"); fig.update_yaxes(gridcolor="#1e293b")
    return fig


def microcycle_chart(d: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure([go.Bar(x=d.index, y=d.planned, name="Planificada", marker_color=SLATE, opacity=.6),
                     go.Bar(x=d.index, y=d.real, name="Real", marker_color=CYAN)])
    fig.update_layout(barmode="group", yaxis_title=label)
    return _style(fig, f"Microciclo (media plantilla): {label} planificada vs real", 360)


def scatter_quadrants(p: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, g in p.groupby("perfil"):
        fig.add_scatter(x=g.nord, y=g.vmax, mode="markers+text", name=name, text=g.name.str[:6] + " (" + g.position + ")",
                        textposition="top center", textfont=dict(size=9),
                        marker=dict(size=13, color=PROFILE_COLORS[name], line=dict(width=1, color="#0f172a")),
                        hovertemplate="<b>%{text}</b><br>NordBord %{x:.2f} N/kg<br>Vmax %{y:.1f} km/h<extra></extra>")
    mx, my = p.nord.median(), p.vmax.median()
    fig.add_vline(x=mx, line_dash="dash", line_color=SLATE); fig.add_hline(y=my, line_dash="dash", line_color=SLATE)
    for x, y, t, xa, ya in [(0, 1, "Velocista con déficit de fuerza (riesgo)", "left", "top"),
                            (1, 1, "Fuerte + rápido", "right", "top"),
                            (0, 0, "Déficit global", "left", "bottom"), (1, 0, "Fuerte, baja velocidad", "right", "bottom")]:
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=t, showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(color=SLATE, size=11))
    fig.update_layout(xaxis_title="Fuerza excéntrica isquios NordBord (N/kg)", yaxis_title="Vmax GPS (km/h)")
    return _style(fig, "Cuadrantes Fuerza–Velocidad", 520)


def radar(cats: list[str], player: list[float], pos_mean: list[float], pname: str, pos: str) -> go.Figure:
    fig = go.Figure()
    for vals, nm, col, op in [(pos_mean, f"Media {pos}", SLATE, .25), (player, pname, CYAN, .35)]:
        fig.add_scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=nm, line_color=col, opacity=.9)
    fig.update_layout(polar=dict(bgcolor="#0f172a", radialaxis=dict(range=[0, 100], gridcolor="#1e293b")))
    return _style(fig, "Perfil vs media posicional (% del máximo de la plantilla)", 440)


def acwr_chart(d: pd.DataFrame, metric: str, label: str, acwr: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d.date, y=d[metric], name=label, marker_color=CYAN, opacity=.75, secondary_y=False)
    fig.add_scatter(x=d.date, y=d[acwr], name="ACWR (EWMA 7/28)", mode="lines+markers",
                    line=dict(color="#f8fafc", width=3), secondary_y=True)
    for y0, y1, col, op, t in [(0, .8, AMBER, .10, "Subentrenamiento"), (.8, 1.3, EMERALD, .16, "Zona óptima"),
                               (1.3, 1.5, AMBER, .10, "Precaución"), (1.5, 2.5, RED, .16, "Riesgo / Spike")]:
        fig.add_shape(type="rect", xref="paper", yref="y2", x0=0, x1=1, y0=y0, y1=y1, fillcolor=col, opacity=op,
                      line_width=0, layer="below")
        fig.add_annotation(xref="paper", yref="y2", x=1, y=(y0 + y1) / 2, text=t, showarrow=False, xanchor="right",
                           font=dict(size=10, color=col))
    fig.update_yaxes(title_text=label, secondary_y=False); fig.update_yaxes(title_text="ACWR", range=[0, 2.5], secondary_y=True, showgrid=False)
    return _style(fig, f"Carga diaria y ACWR – {label}", 480)


def benchmark_chart(labels: list[str], vals: list[float]) -> go.Figure:
    col = [EMERALD if v >= 90 else AMBER if v >= 80 else RED for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=col, text=[f"{v:.0f}%" for v in vals], textposition="inside"))
    fig.add_vline(x=100, line_color="#f8fafc", annotation_text="Baseline 100%"); fig.add_vline(x=90, line_dash="dash", line_color=SLATE)
    fig.update_xaxes(range=[0, 120], title="% de la media pre-lesión")
    return _style(fig, "Benchmark vs baseline pre-lesión", 340)


def pain_chart(log: pd.DataFrame) -> go.Figure:
    fig = go.Figure([go.Scatter(x=log.date, y=log.rpe, name="RPE (0-10)", line=dict(color=CYAN, width=3)),
                     go.Scatter(x=log.date, y=log.pain, name="Dolor post-sesión (0-10)", line=dict(color=AMBER, width=3))])
    fig.update_yaxes(range=[0, 10])
    return _style(fig, "Evolución RPE / Dolor", 340)
