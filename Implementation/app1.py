import math
import numpy as np
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# Config
# ===============================
CSV_PATH = "../Results/Data/Clusters/best_clusters_detailed.csv"
DEFAULT_NCOLS = 3         # start wider panels by default
DEFAULT_NROWS = 2         # rows per page
ROW_HEIGHT = 340          # px per row for readability
TOP_MARGIN = 120          # margin for title/legend

# ===============================
# Data Load + Helpers
# ===============================
def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    required = {"Jlf", "mu", "PP", "Dominance"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    for c in ["Jlf", "mu", "PP"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Dominance"] = df["Dominance"].astype(str).str.strip()
    df = df.dropna(subset=["Jlf", "mu", "PP"]).copy()
    return df

def ensure_cluster_id(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    if "ClusterID_renum" in df.columns:
        out = df.copy()
        out["ClusterID_renum"] = pd.to_numeric(out["ClusterID_renum"], errors="coerce")
        return out

    out = df.copy()
    if "ClusterUID" in out.columns:
        def _renumber(g: pd.DataFrame) -> pd.DataFrame:
            u = g[["ClusterUID"]].drop_duplicates().reset_index(drop=True)
            mapping = {uid: i + 1 for i, uid in enumerate(u["ClusterUID"])}
            g["ClusterID_renum"] = g["ClusterUID"].map(mapping).astype(int)
            return g
        out = out.groupby(["Jlf", "mu", "PP"], group_keys=False).apply(_renumber)
    else:
        def _fallback(g: pd.DataFrame) -> pd.DataFrame:
            u = g.drop_duplicates(subset=[y_col, "Dominance"]).reset_index(drop=True)
            u["_rid"] = np.arange(1, len(u) + 1, dtype=int)
            g = g.merge(u, on=[y_col, "Dominance"], how="left").rename(columns={"_rid": "ClusterID_renum"})
            g["ClusterID_renum"] = g["ClusterID_renum"].astype(int)
            return g
        out = out.groupby(["Jlf", "mu", "PP"], group_keys=False).apply(_fallback)
    return out

df_all = load_data(CSV_PATH)

NUMERIC_CANDIDATES = [
    "Total Cells",
    "PersistenceTime",
    "Centroid_Displacement",
    "Velocity",
    "MeanSize_over_time",
    "StdSize_over_time",
    "ClusterSizeCV_over_time",
    "Phenotypic Index",
    "Entropy",
]
Y_OPTIONS = [c for c in NUMERIC_CANDIDATES if c in df_all.columns]
if not Y_OPTIONS:
    raise ValueError("No plottable numeric columns found in CSV.")

# ===============================
# Figure builder (with pagination)
# ===============================
def build_grid_figure(df: pd.DataFrame, plot_type: str, y_col: str, ncols: int, nrows: int, page: int) -> go.Figure:
    if y_col not in df.columns:
        return go.Figure()

    df = df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[y_col])

    if plot_type == "bar":
        df = ensure_cluster_id(df, y_col)

    combos_all = df[["Jlf", "mu", "PP"]].drop_duplicates().reset_index(drop=True)
    total = len(combos_all)
    per_page = max(1, ncols * nrows)
    max_pages = max(1, math.ceil(total / per_page))
    page = min(max(1, page), max_pages)

    start = (page - 1) * per_page
    end = start + per_page
    combos = combos_all.iloc[start:end].reset_index(drop=True)

    # ---- spacing & fonts
    VSPACING = 0.18          # more room between rows
    HSPACING = 0.05
    ROW_HEIGHT = 360         # px per row
    TOP_MARGIN = 160         # to clear the legend/title
    SUBTITLE_SIZE = 14       # subplot title font size
    AXIS_TITLE_SIZE = 12
    TICK_SIZE = 10

    titles = [f"<b>Jlf={r['Jlf']}, μ={r['mu']}, PP={r['PP']}</b>" for _, r in combos.iterrows()]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        vertical_spacing=VSPACING,
        horizontal_spacing=HSPACING
    )

    leader_color = "#1f77b4"
    follower_color = "#ff7f0e"

    for idx, (_, prm) in enumerate(combos.iterrows(), start=1):
        r = (idx - 1) // ncols + 1
        c = (idx - 1) % ncols + 1
        sub = df[(df["Jlf"] == prm["Jlf"]) & (df["mu"] == prm["mu"]) & (df["PP"] == prm["PP"])].copy()
        if sub.empty:
            continue

        if plot_type == "bar":
            sub = sub.dropna(subset=["ClusterID_renum"]).sort_values(["ClusterID_renum", y_col])
            for dom, col in [("Leader-dominated", leader_color), ("Follower-dominated", follower_color)]:
                s = sub[sub["Dominance"] == dom]
                if s.empty: 
                    continue
                fig.add_trace(
                    go.Bar(
                        x=s["ClusterID_renum"], y=s[y_col],
                        name=dom, marker_color=col,
                        showlegend=(idx == 2),
                    ),
                    row=r, col=c
                )
            fig.update_xaxes(title_text="Cluster ID", row=r, col=c)
            fig.update_yaxes(title_text=y_col, row=r, col=c)

        elif plot_type == "box":
            for dom, col in [("Leader-dominated", leader_color), ("Follower-dominated", follower_color)]:
                s = sub[sub["Dominance"] == dom]
                if s.empty: 
                    continue
                fig.add_trace(
                    go.Box(
                        y=s[y_col], name=dom, marker_color=col,
                        boxmean="sd", showlegend=(idx == 2),
                    ),
                    row=r, col=c
                )
            fig.update_yaxes(title_text=y_col, row=r, col=c)

        elif plot_type == "violin":
            for dom, col in [("Leader-dominated", leader_color), ("Follower-dominated", follower_color)]:
                s = sub[sub["Dominance"] == dom]
                if s.empty: 
                    continue
                fig.add_trace(
                    go.Violin(
                        y=s[y_col], name=dom, fillcolor=col, line_color="black",
                        opacity=0.85, points=False, meanline_visible=True,
                        showlegend=(idx == 2),
                    ),
                    row=r, col=c
                )
            fig.update_yaxes(title_text=y_col, row=r, col=c)

    if plot_type == "bar":
        fig.update_layout(barmode="group")

    # ---- make subplot titles bigger & shifted
    for ann in fig.layout.annotations or []:
        ann.font.size = SUBTITLE_SIZE
        ann.yshift = 8   # nudge titles upward a bit

    # ---- add spacing around axis titles & ticks everywhere
    fig.update_xaxes(title_standoff=14, tickfont_size=TICK_SIZE, title_font_size=AXIS_TITLE_SIZE)
    fig.update_yaxes(title_standoff=10, tickfont_size=TICK_SIZE, title_font_size=AXIS_TITLE_SIZE)

    # ---- legend above, with enough top margin to avoid overlap
    fig.update_layout(
        height=max(700, nrows * ROW_HEIGHT + TOP_MARGIN),
        title=dict(
            text=f"<b>{plot_type.capitalize()} of {y_col}</b> — Page {page}/{max_pages}",
            font=dict(size=18)  # figure title bold via <b>…</b>
        ),
        legend=dict(
            orientation="h",
            y=1.15, yanchor="bottom",   # push legend higher
            x=0.5, xanchor="center",
            font=dict(size=16, family="DejaVu Sans Bold, Arial Black, Arial, sans-serif")
        ),
        margin=dict(l=30, r=30, t=TOP_MARGIN, b=40),
        template="plotly_white",
    )
    return fig, max_pages


# ===============================
# Dash App
# ===============================
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "padding": "12px"},
    children=[
        html.H2("Cluster Metrics Dashboard"),

        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "alignItems": "end"},
            children=[
                html.Div([
                    html.Label("Plot type"),
                    dcc.Dropdown(
                        id="plot-type",
                        options=[{"label": k.capitalize(), "value": k} for k in ["bar", "box", "violin"]],
                        value="bar",
                        clearable=False,
                        style={"width": 200},
                    ),
                ]),
                html.Div([
                    html.Label("Metric (y-axis)"),
                    dcc.Dropdown(
                        id="y-col",
                        options=[{"label": col, "value": col} for col in Y_OPTIONS],
                        value=Y_OPTIONS[0],
                        clearable=False,
                        style={"width": 320},
                    ),
                ]),
                html.Div([
                    html.Label("Grid columns"),
                    html.Div(
                        dcc.Slider(
                            id="n-cols",
                            min=2, max=6, step=1, value=DEFAULT_NCOLS,
                            marks={i: str(i) for i in range(2, 7)},
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="mouseup",
                        ),
                        style={"width": "320px"},
                    ),
                ]),
                html.Div([
                    html.Label("Rows per page"),
                    html.Div(
                        dcc.Slider(
                            id="n-rows",
                            min=1, max=4, step=1, value=DEFAULT_NROWS,
                            marks={i: str(i) for i in range(1, 5)},
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="mouseup",
                        ),
                        style={"width": "320px"},
                    ),
                ]),
                html.Div([
                    html.Label("Page"),
                    html.Div(
                        dcc.Slider(
                            id="page",
                            min=1, max=1, step=1, value=1,   # will be updated dynamically
                            marks={1: "1"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="mouseup",
                        ),
                        style={"width": "360px"},
                    ),
                ]),
            ],
        ),

        html.Div(style={"height": "12px"}),

        # Scrollable graph container
        html.Div(
            dcc.Graph(id="grid-fig"),
            style={
                "height": "85vh",
                "overflowY": "auto",
                "border": "1px solid #eee",
                "padding": "6px",
            },
        ),
    ]
)

# Update page slider (max, marks, value) whenever layout changes
@app.callback(
    Output("page", "max"),
    Output("page", "marks"),
    Output("page", "value"),
    Input("n-cols", "value"),
    Input("n-rows", "value"),
)
def update_page_slider(ncols, nrows):
    combos_total = len(df_all[["Jlf", "mu", "PP"]].drop_duplicates())
    per_page = max(1, int(ncols) * int(nrows))
    max_pages = max(1, math.ceil(combos_total / per_page))
    marks = {i: str(i) for i in range(1, max_pages + 1)}
    return max_pages, marks, min(1, max_pages)

# Build figure for the selected page
@app.callback(
    Output("grid-fig", "figure"),
    Input("plot-type", "value"),
    Input("y-col", "value"),
    Input("n-cols", "value"),
    Input("n-rows", "value"),
    Input("page", "value"),
)
def update_figure(plot_type, y_col, n_cols, n_rows, page):
    df = df_all.copy()
    if y_col not in df.columns:
        return go.Figure()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[y_col])

    fig, _ = build_grid_figure(
        df, plot_type=plot_type, y_col=y_col,
        ncols=int(n_cols), nrows=int(n_rows), page=int(page)
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
