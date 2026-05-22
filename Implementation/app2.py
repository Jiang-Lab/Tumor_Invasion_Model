import math
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt  # for color palettes

# =============================
# CONFIG
# =============================
COMPOSITION_CSV = "../Results/Data/Clusters/ClusterComposition_1.csv"
PARAM_LIST_CSV  = "../Results/Data/Clusters/best_clusters_per_jlf_mu.csv"  # optional
APP_TITLE       = "Cluster Trajectories"

# Fixed scale per selection so full trajectory stays visible while animating
MIN_X_SPAN = 50.0   # min x width (data units)
MIN_Y_SPAN = 5.0    # min y height (data units)
PAD_FRAC   = 0.02   # small padding when data span > min span

# Tracker thresholds
JACCARD_MIN       = 0.15
CENTROID_MAX_DIST = 10.0

# Styling
LINE_WIDTH   = 2
DOT_SIZE     = 7
START_SIZE   = 11   # open circle at start
END_SIZE     = 12   # X at current end
TEXT_FONT_SZ = 10

# =============================
# Helper functions
# =============================
def parse_list(s):
    if pd.isna(s):
        return []
    try:
        v = literal_eval(str(s))
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
    except Exception:
        pass
    return []

def row_points(row, round_dec=1):
    lx = parse_list(row.get("Member_Leader_X"))
    ly = parse_list(row.get("Member_Leader_Y"))
    fx = parse_list(row.get("Member_Follower_X"))
    fy = parse_list(row.get("Member_Follower_Y"))
    pts = [(x, y) for x, y in zip(lx, ly)] + [(x, y) for x, y in zip(fx, fy)]
    if not pts:
        return np.empty((0, 2), dtype=float)
    pts = np.asarray(pts, dtype=float)
    return np.round(pts, round_dec) if round_dec is not None else pts

def jaccard_from_points(A, B):
    if A.size == 0 or B.size == 0:
        return 0.0
    sA, sB = set(map(tuple, A)), set(map(tuple, B))
    inter, uni = len(sA & sB), len(sA | sB)
    return 0.0 if uni == 0 else inter / uni

def centroid_from_points(pts):
    if pts.size == 0:
        return np.nan, np.nan
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

def color_cycle(n):
    """Return n RGBA tuples (0..1) using a sequence of qualitative colormaps."""
    cmaps = ["tab10", "tab20", "tab20b", "tab20c", "Set3", "Accent", "Dark2", "Paired"]
    cols, idx = [], 0
    while len(cols) < n:
        cmap = plt.get_cmap(cmaps[idx % len(cmaps)])
        steps = 20 if "tab20" in cmap.name else 10
        for i in range(steps):
            if len(cols) >= n:
                break
            cols.append(cmap(i / max(1, steps - 1)))
        idx += 1
    return cols[:n]

def mpl_rgba_to_plotly_rgba(c):
    """Convert Matplotlib rgba tuple (0..1) to CSS rgba string."""
    r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
    a = c[3] if len(c) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.3f})"

# =============================
# Core tracker (cumulative trajectories)
# =============================
def build_tracks_df(df, jlf, mu, pp, rep,
                    jaccard_min=JACCARD_MIN, centroid_max_dist=CENTROID_MAX_DIST):
    """
    Build a stable TrackID mapping for a single (Jlf, mu, PP, rep) and
    return DataFrame: MCS, Cluster ID, TrackID, centroid_x_fix, centroid_y_fix, TotalCells
    """
    sub = df[(df["Jlf"] == jlf) & (df["mu"] == mu) & (df["PP"] == pp) & (df["rep"] == rep)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["MCS","Cluster ID","TrackID","centroid_x_fix","centroid_y_fix","TotalCells"])
    sub = sub.sort_values(["MCS", "Cluster ID"]).reset_index(drop=True)

    time_points = sorted(sub["MCS"].unique())
    track_id_counter = 1
    active_tracks = []  # dicts: {'track': int, 'points': Nx2, 'centroid': (x,y)}
    assignments = []

    for t in time_points:
        cur = sub[sub["MCS"] == t].copy()
        cur["pts"] = cur.apply(row_points, axis=1)
        cur["centroid_from_pts"] = cur["pts"].apply(centroid_from_points)

        unmatched = set(cur.index.tolist())
        used_tracks = set()

        # 1) Jaccard match
        for ci in list(unmatched):
            row_pts = cur.at[ci, "pts"]
            best_track, best_j = None, -1.0
            for tr in active_tracks:
                if tr["track"] in used_tracks:
                    continue
                j = jaccard_from_points(row_pts, tr["points"])
                if j > best_j:
                    best_j, best_track = j, tr
            if best_track is not None and best_j >= jaccard_min:
                cx, cy = cur.at[ci, "centroid_from_pts"]
                if np.isnan(cx) or np.isnan(cy):
                    cx, cy = float(cur.at[ci, "Centroid_X"]), float(cur.at[ci, "Centroid_Y"])
                assignments.append({
                    "MCS": t,
                    "Cluster ID": cur.at[ci, "Cluster ID"],
                    "TrackID": best_track["track"],
                    "centroid_x_fix": cx,
                    "centroid_y_fix": cy,
                    "TotalCells": float(cur.at[ci, "Total Cells"]) if "Total Cells" in cur.columns else np.nan
                })
                best_track["points"] = row_pts
                best_track["centroid"] = (cx, cy)
                unmatched.remove(ci)
                used_tracks.add(best_track["track"])

        # 2) Centroid distance match
        for ci in list(unmatched):
            cx, cy = cur.at[ci, "centroid_from_pts"]
            if np.isnan(cx) or np.isnan(cy):
                cx, cy = float(cur.at[ci, "Centroid_X"]), float(cur.at[ci, "Centroid_Y"])
            best_track, best_dist = None, float("inf")
            for tr in active_tracks:
                if tr["track"] in used_tracks:
                    continue
                tx, ty = tr["centroid"]
                if tx is None or np.isnan(tx) or ty is None or np.isnan(ty):
                    continue
                d = math.hypot(cx - tx, cy - ty)
                if d < best_dist:
                    best_dist, best_track = d, tr
            if best_track is not None and best_dist <= centroid_max_dist:
                assignments.append({
                    "MCS": t,
                    "Cluster ID": cur.at[ci, "Cluster ID"],
                    "TrackID": best_track["track"],
                    "centroid_x_fix": cx,
                    "centroid_y_fix": cy,
                    "TotalCells": float(cur.at[ci, "Total Cells"]) if "Total Cells" in cur.columns else np.nan
                })
                best_track["points"] = cur.at[ci, "pts"]
                best_track["centroid"] = (cx, cy)
                unmatched.remove(ci)
                used_tracks.add(best_track["track"])

        # 3) New tracks
        for ci in list(unmatched):
            cx, cy = cur.at[ci, "centroid_from_pts"]
            if np.isnan(cx) or np.isnan(cy):
                cx, cy = float(cur.at[ci, "Centroid_X"]), float(cur.at[ci, "Centroid_Y"])
            tr = {"track": track_id_counter, "points": cur.at[ci, "pts"], "centroid": (cx, cy)}
            active_tracks.append(tr)
            assignments.append({
                "MCS": t,
                "Cluster ID": cur.at[ci, "Cluster ID"],
                "TrackID": track_id_counter,
                "centroid_x_fix": cx,
                "centroid_y_fix": cy,
                "TotalCells": float(cur.at[ci, "Total Cells"]) if "Total Cells" in cur.columns else np.nan
            })
            track_id_counter += 1

    tracks_df = pd.DataFrame(assignments).sort_values(["TrackID","MCS"]).reset_index(drop=True)
    return tracks_df

# =============================
# Load data once
# =============================
df = pd.read_csv(COMPOSITION_CSV)
need = ["MCS","Jlf","mu","PP","rep","Cluster ID",
        "Centroid_X","Centroid_Y",
        "Member_Leader_X","Member_Leader_Y",
        "Member_Follower_X","Member_Follower_Y",
        "Total Cells"]  # needed for per-dot text
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {COMPOSITION_CSV}: {missing}")
for c in ["MCS","Jlf","mu","PP","rep","Cluster ID","Centroid_X","Centroid_Y","Total Cells"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["MCS","Jlf","mu","PP","rep","Cluster ID"]).copy()


USE_BEST_FILE = False  # set to True if you want to use the best_clusters_per_jlf_mu.csv

if USE_BEST_FILE and Path(PARAM_LIST_CSV).exists():
    plist = pd.read_csv(PARAM_LIST_CSV)
    for c in ["Jlf","mu","PP"]:
        plist[c] = pd.to_numeric(plist[c], errors="coerce")
    plist = plist.dropna(subset=["Jlf","mu","PP"]).drop_duplicates(["Jlf","mu","PP"])\
                 .sort_values(["Jlf","mu","PP"])
else:
    plist = df[["Jlf","mu","PP"]].drop_duplicates().sort_values(["Jlf","mu","PP"])
    
    
    

# =============================
# Dash App
# =============================
app = Dash(__name__)
app.title = APP_TITLE

# Layout
app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "padding": "16px"},
    children=[
        html.H2(APP_TITLE, style={"marginBottom": "8px"}),

        html.Div([
            html.Div([
                html.Label("Jlf"),
                dcc.Dropdown(
                    id="dd-jlf",
                    options=[{"label": f"{v:g}", "value": float(v)} for v in sorted(plist["Jlf"].unique())],
                    value=float(sorted(plist["Jlf"].unique())[0]) if not plist.empty else None,
                    clearable=False
                ),
            ], style={"width": "16rem", "marginRight": "1rem"}),

            html.Div([
                html.Label("μ (mu)"),
                dcc.Dropdown(id="dd-mu", clearable=False),
            ], style={"width": "16rem", "marginRight": "1rem"}),

            html.Div([
                html.Label("PP"),
                dcc.Dropdown(id="dd-pp", clearable=False),
            ], style={"width": "16rem", "marginRight": "1rem"}),

            html.Div([
                html.Label("rep"),
                dcc.Dropdown(id="dd-rep", clearable=False),
            ], style={"width": "12rem"}),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "0.5rem"}),

        html.Div(style={"height": "12px"}),

        dcc.Graph(id="traj-graph", style={"height": "72vh"}),

        # Hidden store used by clientside autoplay callback
        dcc.Store(id="autoplay", data=0),

        html.Div(id="status", style={"marginTop": "8px", "color": "#555"})
    ]
)

# Chained dropdowns
@app.callback(
    Output("dd-mu", "options"),
    Output("dd-mu", "value"),
    Input("dd-jlf", "value"),
)
def update_mu(jlf):
    if jlf is None:
        return [], None
    mus = sorted(plist[plist["Jlf"] == jlf]["mu"].unique())
    opts = [{"label": f"{v:g}", "value": float(v)} for v in mus]
    return opts, (float(mus[0]) if mus else None)

@app.callback(
    Output("dd-pp", "options"),
    Output("dd-pp", "value"),
    Input("dd-jlf", "value"),
    Input("dd-mu", "value"),
)
def update_pp(jlf, mu):
    if jlf is None or mu is None:
        return [], None
    pps = sorted(plist[(plist["Jlf"] == jlf) & (plist["mu"] == mu)]["PP"].unique())
    opts = [{"label": f"{v:g}", "value": float(v)} for v in pps]
    default = 0.9 if 0.9 in pps else (float(pps[0]) if pps else None)
    return opts, default

@app.callback(
    Output("dd-rep", "options"),
    Output("dd-rep", "value"),
    Input("dd-jlf", "value"),
    Input("dd-mu", "value"),
    Input("dd-pp", "value"),
)
def update_rep(jlf, mu, pp):
    if None in (jlf, mu, pp):
        return [], None
    reps = sorted(df[(df["Jlf"] == jlf) & (df["mu"] == mu) & (df["PP"] == pp)]["rep"].unique())
    opts = [{"label": str(int(r)) if float(r).is_integer() else f"{r:g}", "value": float(r)} for r in reps]
    return opts, (opts[0]["value"] if opts else None)

# Figure with frames (start circle + end X + colorbar + per-dot TotalCells)
@app.callback(
    Output("traj-graph", "figure"),
    Output("status", "children"),
    Input("dd-jlf", "value"),
    Input("dd-mu", "value"),
    Input("dd-pp", "value"),
    Input("dd-rep", "value"),
)
def update_figure(jlf, mu, pp, rep):
    if None in (jlf, mu, pp, rep):
        return go.Figure(), "Select parameters to view trajectories."

    trdf = build_tracks_df(df, jlf, mu, pp, rep)
    if trdf.empty:
        return go.Figure(), f"No tracks for Jlf={jlf:g}, μ={mu:g}, PP={pp:g}, rep={rep:g}"

    mcs_vals = sorted(trdf["MCS"].unique())
    mcs_min, mcs_max = float(min(mcs_vals)), float(max(mcs_vals))
    tids     = list(trdf["TrackID"].unique())
    cols     = [mpl_rgba_to_plotly_rgba(c) for c in color_cycle(len(tids))]
    per_tid  = {tid: trdf[trdf["TrackID"] == tid].sort_values("MCS") for tid in tids}

    # Fixed limits so full trajectory remains visible
    xmin = float(trdf["centroid_x_fix"].min()); xmax = float(trdf["centroid_x_fix"].max())
    ymin = float(trdf["centroid_y_fix"].min()); ymax = float(trdf["centroid_y_fix"].max())
    dx = max(xmax - xmin, 0.0); dy = max(ymax - ymin, 0.0)
    eff_x_span = max(MIN_X_SPAN, dx * (1.0 + PAD_FRAC))
    eff_y_span = max(MIN_Y_SPAN, dy * (1.0 + PAD_FRAC))
    x_center = 0.5 * (xmax + xmin) if dx > 0 else xmin
    y_center = 0.5 * (ymax + ymin) if dy > 0 else ymin
    xlim = (x_center - eff_x_span / 2.0, x_center + eff_x_span / 2.0)
    ylim = (y_center - eff_y_span / 2.0, y_center + eff_y_span / 2.0)

    fig = go.Figure()

    # --- Base traces: for each cluster, add 4 traces (line, dots, start, end) ---
    trace_order = []  # (tid, role) roles: "line", "dots", "start", "end"
    for idx, (tid, col) in enumerate(zip(tids, cols)):
        g = per_tid[tid]
        # 1) LINE (updates each frame)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines", name=f"Cluster {tid}",
            line=dict(width=LINE_WIDTH, color=col),
            showlegend=True
        ))
        trace_order.append((tid, "line"))

        # 2) DOTS colored by MCS with text = TotalCells (updates each frame)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers+text", text=[], textposition="top center",
            marker=dict(size=DOT_SIZE, color=[], colorscale="Viridis",
                        cmin=mcs_min, cmax=mcs_max,
                        showscale=(idx == 0), colorbar=dict(title="MCS") if idx == 0 else None),
            name=f"Cluster {tid} points",
            showlegend=False
        ))
        trace_order.append((tid, "dots"))

        # 3) START marker (fixed at first point; open circle)
        if not g.empty:
            x0, y0 = float(g.iloc[0]["centroid_x_fix"]), float(g.iloc[0]["centroid_y_fix"])
        else:
            x0, y0 = None, None
        fig.add_trace(go.Scatter(
            x=[x0] if x0 is not None else [], y=[y0] if y0 is not None else [],
            mode="markers", name=f"Cluster {tid} start",
            marker=dict(symbol="circle-open", size=START_SIZE, line=dict(width=2, color=col)),
            showlegend=False
        ))
        trace_order.append((tid, "start"))

        # 4) END marker (dynamic X at current last point)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers", name=f"Cluster {tid} end",
            marker=dict(symbol="x", size=END_SIZE, line=dict(width=2, color=col), color=col),
            showlegend=False
        ))
        trace_order.append((tid, "end"))
    # ---------------------------------------------------------------------------

    # Frames: update line, dots (+ text, color), and end marker (guard for empty gk)
    frames = []
    for mcs_k in mcs_vals:
        frame_data = []
        for tid, role in trace_order:
            g = per_tid[tid]
            # Placeholder if no rows for this cluster at all
            if g.empty:
                frame_data.append(go.Scatter(x=[], y=[], mode="markers"))
                continue

            gk = g[g["MCS"] <= mcs_k]
            if role == "line":
                frame_data.append(go.Scatter(
                    x=gk["centroid_x_fix"], y=gk["centroid_y_fix"],
                    mode="lines"
                ))
            elif role == "dots":
                frame_data.append(go.Scatter(
                    x=gk["centroid_x_fix"], y=gk["centroid_y_fix"],
                    mode="markers+text",
                    marker=dict(
                        size=DOT_SIZE,
                        color=gk["MCS"].to_numpy(), colorscale="Viridis",
                        cmin=mcs_min, cmax=mcs_max,
                        showscale=False  # colorbar on base only
                    ),
                    text=[f'{int(tc)}' if not pd.isna(tc) else '' for tc in gk["TotalCells"]],
                    textfont=dict(size=TEXT_FONT_SZ),
                    textposition="top center"
                ))
            elif role == "start":
                # Start marker is static; no update per frame
                frame_data.append(go.Scatter(x=[], y=[], mode="markers"))
            elif role == "end":
                if gk.empty:
                    # Track hasn't started by this frame -> no end marker yet
                    frame_data.append(go.Scatter(x=[], y=[], mode="markers"))
                else:
                    xe = float(gk["centroid_x_fix"].iloc[-1])
                    ye = float(gk["centroid_y_fix"].iloc[-1])
                    frame_data.append(go.Scatter(
                        x=[xe], y=[ye],
                        mode="markers",
                        marker=dict(symbol="x", size=END_SIZE)
                    ))
        frames.append(go.Frame(data=frame_data, name=str(mcs_k)))

    # Layout + controls
    slider_steps = [
        dict(label=str(m), method="animate",
             args=[[str(m)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}])
        for m in mcs_vals
    ]
    fig.update_layout(
        title=f"Trajectories — Jlf={jlf:g}, μ={mu:g}, PP={pp:g}, rep={rep:g}",
        xaxis=dict(title="Centroid_X", range=[xlim[0], xlim[1]]),
        yaxis=dict(title="Centroid_Y", range=[ylim[0], ylim[1]]),
        legend=dict(orientation="h", yanchor="bottom", y=0.95, xanchor="left", x=0.0),
        margin=dict(l=50, r=20, t=60, b=40),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            x=1.0, xanchor="right",
            y=1.15, yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"fromcurrent": True,
                                  "frame": {"duration": 300, "redraw": True},
                                  "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False}}]),
                dict(label="⟲ Reset", method="animate",
                     args=[[str(mcs_vals[0])], {"mode": "immediate",
                                                "frame": {"duration": 0, "redraw": True}}]),
            ]
        )],
        sliders=[dict(
            active=0,
            y=-0.07, x=0.05, len=0.9,
            pad=dict(t=0, b=0),
            steps=slider_steps
        )],
        template="plotly_white",
    )
    fig.frames = frames

    status = f"{len(tids)} clusters • {len(mcs_vals)} time points • limits: x={xlim[0]:.2f}..{xlim[1]:.2f}, y={ylim[0]:.2f}..{ylim[1]:.2f}"
    return fig, status

# =============================
# Client-side autoplay
# =============================
app.clientside_callback(
    """
    function(fig){
        if(!fig || !fig.frames || !fig.frames.length){
            return window.dash_clientside.no_update;
        }
        var host = document.getElementById('traj-graph');
        if(!host) return window.dash_clientside.no_update;
        var gd = host.getElementsByClassName('js-plotly-plot')[0] || host;
        try {
            Plotly.animate(
                gd,
                null,  // play all frames in order
                {frame: {duration: 300, redraw: true},
                 transition: {duration: 0},
                 fromcurrent: true}
            );
        } catch(e) {}
        return 1; // dummy value
    }
    """,
    Output("autoplay", "data"),
    Input("traj-graph", "figure")
)

if __name__ == "__main__":
    app.run(debug=True)

