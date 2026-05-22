import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Load data
df_reps = pd.read_csv("../Results/Cluster/ClusterRate_CFR.csv")
df_avg = pd.read_csv("../Results/Cluster/Average_ClusterRate_CFR.csv")
df_cv_reps = pd.read_csv("../Results/Cluster/ClusterSizeCV.csv")
df_cv_avg = pd.read_csv("../Results/Cluster/ClusterSizeCV_Avg.csv")

# Precompute summary for heatmap
summary_stats = (
    df_avg.groupby(['Jlf', 'mu', 'PP'])
    .agg({'CFR': 'mean'})
    .reset_index()
    .rename(columns={'CFR': 'CFR_mean'})
)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Tumor Cluster Dynamics Dashboard"

# Layout
app.layout = html.Div([
    html.H2("Tumor Cluster Dynamics Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Jlf:"),
        dcc.Dropdown(
            options=[{"label": str(v), "value": v} for v in sorted(df_reps["Jlf"].unique())],
            id="jlf-select", value=df_reps["Jlf"].unique()[0]
        ),
        html.Label("Select mu:"),
        dcc.Dropdown(
            options=[{"label": str(v), "value": v} for v in sorted(df_reps["mu"].unique())],
            id="mu-select", value=df_reps["mu"].unique()[0]
        ),
        html.Label("Select PP:"),
        dcc.Dropdown(
            options=[{"label": str(v), "value": v} for v in sorted(df_reps["PP"].unique())],
            id="pp-select", value=df_reps["PP"].unique()[0]
        ),
    ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

    html.Div([
        dcc.Graph(id="replicate-subplots"),
        dcc.Graph(id="average-lineplot"),
        dcc.Graph(id="cv-replicate-subplots"),
        dcc.Graph(id="cv-average-lineplot"),
        dcc.Graph(id="heatmap")
    ], style={"width": "70%", "display": "inline-block", "padding": "10px"}),
])

# Callback to update plots
@app.callback(
    Output("replicate-subplots", "figure"),
    Output("average-lineplot", "figure"),
    Output("cv-replicate-subplots", "figure"),
    Output("cv-average-lineplot", "figure"),
    Output("heatmap", "figure"),
    Input("jlf-select", "value"),
    Input("mu-select", "value"),
    Input("pp-select", "value"),
)
def update_plots(jlf, mu, pp):
    # === Cluster Rate + CFR Replicates ===
    df_filtered = df_reps[(df_reps["Jlf"] == jlf) & (df_reps["mu"] == mu) & (df_reps["PP"] == pp)]
    rep_ids = sorted(df_filtered["rep"].unique())
    n_reps = len(rep_ids)
    n_cols = min(4, n_reps)
    n_rows = math.ceil(n_reps / n_cols)

    fig_sub = make_replicate_subplots(df_filtered, rep_ids, n_rows, n_cols)

    # === Cluster Rate + CFR Average ===
    df_avg_filtered = df_avg[(df_avg["Jlf"] == jlf) & (df_avg["mu"] == mu) & (df_avg["PP"] == pp)]
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=df_avg_filtered["MCS"], y=df_avg_filtered["Cluster_Rate"],
                                 mode='lines+markers', name='Avg Cluster Rate'))
    fig_avg.add_trace(go.Scatter(x=df_avg_filtered["MCS"], y=df_avg_filtered["CFR"],
                                 mode='lines+markers', name='Avg CFR'))
    fig_avg.update_layout(title="Average Cluster Rate and CFR", xaxis_title="MCS", yaxis_title="Rate")

    # === CV per Replicate ===
    df_cv_filtered = df_cv_reps[(df_cv_reps["Jlf"] == jlf) & (df_cv_reps["mu"] == mu) & (df_cv_reps["PP"] == pp)]
    rep_ids_cv = sorted(df_cv_filtered["rep"].unique())
    n_reps_cv = len(rep_ids_cv)
    n_cols_cv = min(4, n_reps_cv)
    n_rows_cv = math.ceil(n_reps_cv / n_cols_cv)

    fig_cv_sub = make_cv_subplots(df_cv_filtered, rep_ids_cv, n_rows_cv, n_cols_cv)

    # === CV Averaged ===
    df_cv_avg_filtered = df_cv_avg[(df_cv_avg["Jlf"] == jlf) & (df_cv_avg["mu"] == mu) & (df_cv_avg["PP"] == pp)]
    fig_cv_avg = go.Figure()
    fig_cv_avg.add_trace(go.Scatter(x=df_cv_avg_filtered["MCS"], y=df_cv_avg_filtered["Avg CV"],
                                    mode='lines+markers', name='Avg CV', line=dict(color="red")))
    fig_cv_avg.update_layout(title="Average CV of Cluster Size", xaxis_title="MCS", yaxis_title="CV")

    # === Heatmap ===
    heat_df = summary_stats[summary_stats["PP"] == pp]
    pivot = heat_df.pivot(index="Jlf", columns="mu", values="CFR_mean")
    fig_heat = px.imshow(pivot, labels={"color": "CFR_mean"},
                         title=f"CFR Mean Heatmap (PP={pp})", aspect="auto")

    return fig_sub, fig_avg, fig_cv_sub, fig_cv_avg, fig_heat

# Helper: Cluster Rate/CFR subplots
def make_replicate_subplots(df, rep_ids, rows, cols):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Rep {r}" for r in rep_ids])
    for i, rep in enumerate(rep_ids):
        row = i // cols + 1
        col = i % cols + 1
        sub_df = df[df["rep"] == rep]
        fig.add_trace(go.Scatter(x=sub_df["MCS"], y=sub_df["Cluster_Rate"], name=f"CR Rep {rep}",
                                 mode='lines', line=dict(color="blue")), row=row, col=col)
        fig.add_trace(go.Scatter(x=sub_df["MCS"], y=sub_df["CFR"], name=f"CFR Rep {rep}",
                                 mode='lines', line=dict(color="orange")), row=row, col=col)
    fig.update_layout(height=250 * rows, width=350 * cols, title_text="Cluster Rate & CFR per Replicate")
    return fig

# Helper: CV subplots
def make_cv_subplots(df, rep_ids, rows, cols):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Rep {r}" for r in rep_ids])
    for i, rep in enumerate(rep_ids):
        row = i // cols + 1
        col = i % cols + 1
        sub_df = df[df["rep"] == rep]
        fig.add_trace(go.Scatter(x=sub_df["MCS"], y=sub_df["CV"], name=f"CV Rep {rep}",
                                 mode='lines+markers', marker=dict(size=4), line=dict(color="green")),
                      row=row, col=col)
    fig.update_layout(height=250 * rows, width=350 * cols, title_text="Coefficient of Variation of Cluster Size per Replicate")
    return fig

# Run app
if __name__ == "__main__":
    app.run(debug=True)






