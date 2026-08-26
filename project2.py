"""
Chicago traffic crashes -- interactive Dash/Plotly dashboard.

A Dash re-build of the "When crashes happen" artifact: filter 50,000 Chicago
crash records by year and weather condition, optionally exclude incomplete
trailing months (reporting lag), and see how counts and injury risk shift
across month, hour, day-of-week, weather, and crash type.

Install once:
    pip install dash plotly pandas

Run:
    python project2.py
Then open http://127.0.0.1:8050 in a browser.
"""

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ---------------------------------------------------------------------------
# Config / palette
# ---------------------------------------------------------------------------
INPUT_PATH = "chicago_crashes_recent_sample_50000.csv"
MIN_CRASH_TYPE_COUNT = 50   # crash types below this get folded into "Other"
N_WEATHER_BUCKETS = 5       # top-N raw weather values kept as-is; rest -> "Other"

SEQ_BLUE = [
    [0.00, "#eef4fc"], [0.15, "#cde2fb"], [0.30, "#9ec5f4"],
    [0.45, "#6da7ec"], [0.60, "#3987e5"], [0.75, "#2a78d6"],
    [0.90, "#1c5cab"], [1.00, "#0d366b"],
]
ACCENT = "#2a78d6"
TRACK = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

BASE_LAYOUT = dict(
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
    font=dict(color=INK_PRIMARY, family=FONT),
)

# ---------------------------------------------------------------------------
# Load & prep data (once, at startup)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
dt = pd.to_datetime(df["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p")

df["crash_year"] = dt.dt.year
df["crash_year_month"] = dt.dt.to_period("M").astype(str)
df["crash_date_only"] = dt.dt.date
df["day_name"] = dt.dt.day_name()
df["hour"] = dt.dt.hour

all_years = sorted(df["crash_year"].unique().tolist())

# -- weather bucketing: keep the top-N raw categories, fold the rest into "Other"
weather_counts = df["WEATHER_CONDITION"].value_counts()
top_weather = weather_counts.head(N_WEATHER_BUCKETS).index.tolist()
WEATHER_ORDER = [w.title() for w in top_weather] + ["Other"]

def bucket_weather(w):
    return w.title() if w in top_weather else "Other"

df["weather_bucket"] = df["WEATHER_CONDITION"].apply(bucket_weather)

# -- crash-type bucketing: fold rare types into "Other", title-case the rest
ct_counts = df["FIRST_CRASH_TYPE"].value_counts()
rare_types = set(ct_counts[ct_counts < MIN_CRASH_TYPE_COUNT].index)

def bucket_crash_type(t):
    return "Other" if t in rare_types else t.title().replace(" To ", " to ")

df["crash_type"] = df["FIRST_CRASH_TYPE"].apply(bucket_crash_type)
df["any_injury"] = (~df["MOST_SEVERE_INJURY"].isin(["NO INDICATION OF INJURY"])) & df["MOST_SEVERE_INJURY"].notna()
df["fatal"] = df["MOST_SEVERE_INJURY"].eq("FATAL")

CT_ORDER = (
    df.groupby("crash_type").size().sort_values(ascending=False).index.tolist()
)
if "Other" in CT_ORDER:
    CT_ORDER = [c for c in CT_ORDER if c != "Other"] + ["Other"]

# -- auto-detect incomplete trailing months (reporting-lag heuristic):
#    a trailing month is "incomplete" if its count drops below 50% of the
#    prior 3-month average; walk back from the most recent month while that
#    holds, so a real slow month deep in the series doesn't get excluded.
ym_sorted = sorted(df["crash_year_month"].unique())
monthly_all = df.groupby("crash_year_month").size().reindex(ym_sorted)
is_low = {}
for i, ym in enumerate(ym_sorted):
    if i < 3:
        is_low[ym] = False
        continue
    prior_avg = monthly_all.iloc[i - 3:i].mean()
    is_low[ym] = bool(monthly_all.iloc[i] < 0.5 * prior_avg)
INCOMPLETE_YM = []
for ym in reversed(ym_sorted):
    if is_low[ym]:
        INCOMPLETE_YM.append(ym)
    else:
        break
INCOMPLETE_YM = sorted(INCOMPLETE_YM)

DATE_RANGE = (str(df["crash_date_only"].min()), str(df["crash_date_only"].max()))
TOTAL_ROWS = len(df)

# ---------------------------------------------------------------------------
# Figure builders -- each takes an already-filtered dataframe
# ---------------------------------------------------------------------------

def empty_figure(msg="No data for this selection"):
    fig = go.Figure()
    fig.update_layout(**BASE_LAYOUT,
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       annotations=[dict(text=msg, showarrow=False,
                                          font=dict(color=INK_MUTED))])
    return fig


def build_trend(d):
    if d.empty:
        return empty_figure()
    monthly = d.groupby("crash_year_month").size().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values,
        mode="lines", fill="tozeroy",
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(42,120,214,0.12)",
        hovertemplate="%{x}<br><b>%{y:,} crashes</b><extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**BASE_LAYOUT,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(showgrid=False, tickangle=-45, color=INK_MUTED, tickfont=dict(size=10)),
        yaxis=dict(title="Crashes", gridcolor=GRIDLINE, zeroline=False, color=INK_MUTED),
    )
    return fig


def build_hour_bar(d):
    if d.empty:
        return empty_figure()
    by_hour = d.groupby("hour").size().reindex(range(24), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_hour.index, y=by_hour.values,
        marker=dict(color=ACCENT),
        hovertemplate="Hour %{x}:00<br><b>%{y:,} crashes</b><extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**BASE_LAYOUT,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="Hour", dtick=2, showgrid=False, color=INK_MUTED),
        yaxis=dict(title="Crashes", gridcolor=GRIDLINE, zeroline=False, color=INK_MUTED),
    )
    return fig


def build_weather_bar(d):
    if d.empty:
        return empty_figure()
    counts = d["weather_bucket"].value_counts().reindex(WEATHER_ORDER, fill_value=0)
    counts = counts.sort_values(ascending=True)  # ascending: Plotly draws bottom-to-top
    total = counts.sum()
    pct = (counts / total * 100) if total else counts * 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=ACCENT),
        text=[f"{p:.0f}%" for p in pct],
        textposition="outside",
        textfont=dict(color=INK_MUTED, size=11),
        customdata=pct.values,
        hovertemplate="<b>%{y}</b><br>%{x:,} crashes (%{customdata:.1f}% of selection)<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**BASE_LAYOUT,
        margin=dict(l=100, r=40, t=10, b=40),
        xaxis=dict(title="Crashes", gridcolor=GRIDLINE, zeroline=False, color=INK_MUTED,
                   range=[0, counts.max() * 1.25 if counts.max() else 1]),
        yaxis=dict(color=INK_PRIMARY, automargin=True),
    )
    return fig


def build_heatmap(d):
    if d.empty:
        return empty_figure()
    pivot = d.pivot_table(index="day_name", columns="hour",
                           values="crash_year", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=DAY_ORDER, fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(range(24)), y=DAY_ORDER,
        colorscale=SEQ_BLUE,
        colorbar=dict(title="Crashes", outlinewidth=0, tickfont=dict(color=INK_MUTED)),
        hovertemplate="%{y}, %{x}:00<br><b>%{z:,} crashes</b><extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(**BASE_LAYOUT,
        margin=dict(l=90, r=20, t=10, b=40),
        xaxis=dict(title="Hour of day", dtick=2, color=INK_MUTED),
        yaxis=dict(color=INK_MUTED, autorange="reversed"),
    )
    return fig


def build_injury_chart(d):
    if d.empty:
        return empty_figure()
    summary = d.groupby("crash_type").agg(
        n=("crash_type", "size"),
        injured=("any_injury", "sum"),
        fatal=("fatal", "sum"),
    )
    summary = summary[summary["n"] > 0]
    if summary.empty:
        return empty_figure()
    summary["rate"] = summary["injured"] / summary["n"] * 100
    summary = summary.sort_values("rate", ascending=True)  # ascending: bottom-to-top

    fig = go.Figure()
    # background "track" so every bar reads against a full-width scale
    fig.add_trace(go.Bar(
        x=[summary["rate"].max() * 1.05] * len(summary),
        y=summary.index, orientation="h",
        marker=dict(color=TRACK), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=summary["rate"], y=summary.index, orientation="h",
        marker=dict(color=ACCENT),
        text=[f"{r:.1f}%" for r in summary["rate"]],
        textposition="outside",
        textfont=dict(color=INK_MUTED, family="IBM Plex Mono, monospace", size=12),
        customdata=summary[["n", "fatal"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>%{x:.1f}% injury rate<br>"
            "%{customdata[0]:,} crashes &middot; %{customdata[1]:,} fatal<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        barmode="overlay", **BASE_LAYOUT,
        margin=dict(l=170, r=60, t=10, b=40),
        xaxis=dict(title="Share of crashes with a reported injury", ticksuffix="%",
                   showgrid=False, zeroline=False, color=INK_MUTED,
                   range=[0, summary["rate"].max() * 1.18]),
        yaxis=dict(showgrid=False, color=INK_PRIMARY, automargin=True),
        height=90 + 32 * len(summary),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "When crashes happen"

CHIP_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "6px"}

_INDEX_CSS = f"""
        body {{ margin: 0; background: {PAGE}; }}
        .chip-check input[type="checkbox"] {{ display: none; }}
        .chip-check label {{
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 13px; padding: 6px 12px; border-radius: 999px;
            border: 1px solid rgba(11,11,11,0.14); background: transparent;
            color: {INK_SECONDARY}; cursor: pointer; user-select: none;
            transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
        }}
        .chip-check label:hover {{ border-color: {ACCENT}; }}
        .chip-check input[type="checkbox"]:checked + label {{
            background: {ACCENT}; color: #ffffff; border-color: {ACCENT}; font-weight: 600;
        }}
        .lag-switch input[type="checkbox"] {{ transform: scale(1.2); margin-right: 8px; }}
        .kpi-card {{
            background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
            border-radius: 10px; padding: 12px 18px; min-width: 150px; flex: 1 1 150px;
        }}
        .kpi-label {{ font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: {INK_MUTED}; }}
        .kpi-value {{ font-size: 21px; font-weight: 700; margin-top: 4px; color: {INK_PRIMARY}; }}
"""

app.index_string = (
    "<!DOCTYPE html>\n<html>\n<head>\n"
    "    {%metas%}\n    <title>{%title%}</title>\n    {%favicon%}\n    {%css%}\n"
    "    <style>" + _INDEX_CSS + "    </style>\n"
    "</head>\n<body>\n    {%app_entry%}\n"
    "    <footer>\n        {%config%}\n        {%scripts%}\n        {%renderer%}\n    </footer>\n"
    "</body>\n</html>"
)


def chart_card(title, desc, graph_id):
    return html.Div(style={
        "background": SURFACE, "border": "1px solid rgba(11,11,11,0.10)",
        "borderRadius": "12px", "padding": "18px 20px", "flex": "1 1 380px",
    }, children=[
        html.H2(title, style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P(desc, style={"margin": "0 0 6px", "fontSize": "12.5px", "color": INK_MUTED}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False}),
    ])


app.layout = html.Div(style={
    "fontFamily": FONT, "backgroundColor": PAGE, "color": INK_PRIMARY,
    "maxWidth": "1180px", "margin": "0 auto", "padding": "40px 24px 80px",
}, children=[

    html.P("Chicago Traffic Crashes · 2021–2026 sample", style={
        "fontFamily": "IBM Plex Mono, monospace", "fontSize": "12px",
        "letterSpacing": "0.08em", "textTransform": "uppercase",
        "color": ACCENT, "margin": "0 0 10px",
    }),
    html.H1("When crashes happen", style={"fontSize": "36px", "margin": "0 0 10px"}),
    html.P(
        "50,000 Chicago Police Department crash records, broken down by year, day of week, "
        "hour, weather condition, and crash type — filter below to see how the rhythm "
        "and risk of crashes shifts.",
        style={"color": INK_SECONDARY, "fontSize": "15px", "maxWidth": "62ch", "margin": "0 0 10px"},
    ),
    html.Span(f"{DATE_RANGE[0]} → {DATE_RANGE[1]}  ·  {TOTAL_ROWS:,} records", style={
        "fontFamily": "IBM Plex Mono, monospace", "fontSize": "12px", "color": INK_MUTED,
    }),

    # Controls
    html.Div(style={
        "background": SURFACE, "border": "1px solid rgba(11,11,11,0.10)", "borderRadius": "12px",
        "display": "flex", "flexWrap": "wrap", "gap": "20px 32px",
        "padding": "18px 20px", "margin": "24px 0 18px",
    }, children=[
        html.Div([
            html.Label("Years", style={
                "display": "block", "fontSize": "11px", "letterSpacing": "0.06em",
                "textTransform": "uppercase", "color": INK_MUTED, "marginBottom": "8px", "fontWeight": 600,
            }),
            dcc.Checklist(
                id="year-filter", className="chip-check",
                options=[{"label": str(y), "value": y} for y in all_years],
                value=all_years, style=CHIP_STYLE,
            ),
        ]),
        html.Div([
            html.Label("Weather", style={
                "display": "block", "fontSize": "11px", "letterSpacing": "0.06em",
                "textTransform": "uppercase", "color": INK_MUTED, "marginBottom": "8px", "fontWeight": 600,
            }),
            dcc.Checklist(
                id="weather-filter", className="chip-check",
                options=[{"label": w, "value": w} for w in WEATHER_ORDER],
                value=WEATHER_ORDER, style=CHIP_STYLE,
            ),
        ]),
        html.Div([
            html.Label("Data quality", style={
                "display": "block", "fontSize": "11px", "letterSpacing": "0.06em",
                "textTransform": "uppercase", "color": INK_MUTED, "marginBottom": "8px", "fontWeight": 600,
            }),
            dcc.Checklist(
                id="lag-filter", className="lag-switch",
                options=[{"label": " Exclude incomplete trailing months", "value": "exclude"}],
                value=["exclude"],
            ),
            html.P(
                ("Currently excludes " + ", ".join(INCOMPLETE_YM) +
                 " — crash counts trail off sharply in these months, a reporting-lag "
                 "artifact rather than a real drop.") if INCOMPLETE_YM else
                "No incomplete trailing months detected in this extract.",
                style={"fontSize": "12px", "color": INK_MUTED, "maxWidth": "320px", "marginTop": "6px"},
            ),
        ]),
    ]),

    # KPI row
    html.Div(id="kpi-row", style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "margin": "0 0 20px"}),

    # Chart grid
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}, children=[
        chart_card("Crashes per month", "Monthly totals across the selected years", "trend-chart"),
        chart_card("Crashes by hour of day", "Summed across every selected year", "hour-chart"),
        chart_card("Crashes by weather condition",
                   "Clear weather dominates simply because most driving happens in clear "
                   "weather — toggle it off above to compare the rest", "weather-chart"),
    ]),

    # Heatmap
    html.Div(style={
        "background": SURFACE, "border": "1px solid rgba(11,11,11,0.10)", "borderRadius": "12px",
        "padding": "18px 20px", "margin": "16px 0",
    }, children=[
        html.H2("Crashes by day & hour", style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P("Where day-of-week and time-of-day compound",
               style={"margin": "0 0 6px", "fontSize": "12.5px", "color": INK_MUTED}),
        dcc.Graph(id="heat-chart", config={"displayModeBar": False}),
    ]),

    # Injury rate
    html.Div(style={
        "background": SURFACE, "border": "1px solid rgba(11,11,11,0.10)", "borderRadius": "12px",
        "padding": "18px 20px", "margin": "16px 0",
    }, children=[
        html.H2("Injury rate by crash type", style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P("Share of crashes with a reported injury, by first crash type",
               style={"margin": "0 0 6px", "fontSize": "12.5px", "color": INK_MUTED}),
        dcc.Graph(id="injury-chart", config={"displayModeBar": False}),
    ]),

    html.Footer(style={
        "background": SURFACE, "border": "1px solid rgba(11,11,11,0.10)", "borderRadius": "12px",
        "padding": "16px 20px", "marginTop": "22px", "fontSize": "12.5px",
        "color": INK_MUTED, "lineHeight": "1.6",
    }, children=[
        html.Strong("Reading this data: ", style={"color": INK_SECONDARY}),
        "the underlying dataset is CPD's E-Crash system, mirrored on the Chicago Data Portal. "
        "Crash reports are added and finalized over time, so the most recent months are "
        "typically undercounted at the point of any given extract — that's why trailing "
        "months can be excluded above. Weather is grouped into its five most common categories "
        "plus Other; crash types under 50 records are folded the same way. \"Injury rate\" "
        "counts any crash with a reported injury (non-incapacitating through fatal); "
        "fatalities are also shown separately. All charts respond to the year, weather, "
        "and data-quality controls together.",
    ]),
])


def kpi_card(label, value):
    return html.Div(className="kpi-card", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ])


# ---------------------------------------------------------------------------
# Callback: one control set drives every chart + the KPI row
# ---------------------------------------------------------------------------
@app.callback(
    Output("trend-chart", "figure"),
    Output("hour-chart", "figure"),
    Output("weather-chart", "figure"),
    Output("heat-chart", "figure"),
    Output("injury-chart", "figure"),
    Output("kpi-row", "children"),
    Input("year-filter", "value"),
    Input("weather-filter", "value"),
    Input("lag-filter", "value"),
)
def update_dashboard(selected_years, selected_weather, lag_value):
    # never allow the selection to go empty -- fall back to "all"
    selected_years = selected_years or all_years
    selected_weather = selected_weather or WEATHER_ORDER

    d = df[df["crash_year"].isin(selected_years) & df["weather_bucket"].isin(selected_weather)]
    if lag_value and "exclude" in lag_value:
        d = d[~d["crash_year_month"].isin(INCOMPLETE_YM)]

    n = len(d)
    if n == 0:
        return (empty_figure(), empty_figure(), empty_figure(), empty_figure(), empty_figure(),
                [kpi_card("Crashes in selection", "0")])

    n_months = d["crash_year_month"].nunique()
    avg_per_day = n / (n_months * 30.44) if n_months else 0
    busiest_day = d.groupby("day_name").size().reindex(DAY_ORDER, fill_value=0).idxmax()
    busiest_hour = int(d.groupby("hour").size().idxmax())
    top_weather = d["weather_bucket"].value_counts().idxmax()

    ct_summary = d.groupby("crash_type").agg(n=("crash_type", "size"), injured=("any_injury", "sum"))
    ct_summary = ct_summary[ct_summary["n"] >= 30]
    if not ct_summary.empty:
        riskiest = (ct_summary["injured"] / ct_summary["n"]).idxmax()
    else:
        riskiest = "—"

    kpis = [
        kpi_card("Crashes in selection", f"{n:,}"),
        kpi_card("Avg crashes / day", f"{avg_per_day:.1f}"),
        kpi_card("Busiest day", busiest_day),
        kpi_card("Busiest hour", f"{busiest_hour:02d}:00"),
        kpi_card("Top weather condition", top_weather),
        kpi_card("Highest-risk crash type", riskiest),
    ]

    return (
        build_trend(d),
        build_hour_bar(d),
        build_weather_bar(d),
        build_heatmap(d),
        build_injury_chart(d),
        kpis,
    )


if __name__ == "__main__":
    app.run(debug=True)
    # If you're on an older Dash version (<2.9) and the line above errors,
    # use app.run_server(debug=True) instead.
