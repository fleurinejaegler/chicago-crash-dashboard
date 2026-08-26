from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

INJURY_THRESHOLD = 0.5
SEVERE_THRESHOLD = 0.5

C = {
    "surface": "#fcfcfb",
    "panel": "#ffffff",
    "border": "#e4e3df",
    "grid": "#eeeeea",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "ink3": "#84837d",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif"

MAP_STYLE = os.environ.get("MAP_STYLE", "carto-positron")

print("loading models and data ...")
M_INJURY = joblib.load(MODEL_DIR / "chicago_injury_lr.joblib")
M_SEVERE = joblib.load(MODEL_DIR / "chicago_severe_lr.joblib")

DF = pd.read_csv(ROOT / "data" / "chicago_crashes_recent_sample_50000.csv")
DF = DF[DF["INJURIES_TOTAL"].notna() & DF["MOST_SEVERE_INJURY"].notna()].copy()
DF["INJURY"] = (DF["INJURIES_TOTAL"] > 0).astype(int)
DF["SEVERE"] = DF["MOST_SEVERE_INJURY"].isin(["INCAPACITATING INJURY", "FATAL"]).astype(int)

BASE_INJURY = float(DF["INJURY"].mean())
BASE_SEVERE = float(DF["SEVERE"].mean())

NUMERIC = ["POSTED_SPEED_LIMIT", "NUM_UNITS", "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH"]
CATEGORICAL = ["WEATHER_CONDITION", "LIGHTING_CONDITION", "ROADWAY_SURFACE_COND",
               "FIRST_CRASH_TYPE", "TRAFFICWAY_TYPE", "PRIM_CONTRIBUTORY_CAUSE"]

DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
        5: "Thursday", 6: "Friday", 7: "Saturday"}

NICE_NUM = {"POSTED_SPEED_LIMIT": "Speed limit", "NUM_UNITS": "Units involved",
            "CRASH_HOUR": "Hour of day", "CRASH_DAY_OF_WEEK": "Day of week",
            "CRASH_MONTH": "Month"}
NICE_CAT = {"WEATHER_CONDITION": "Weather", "LIGHTING_CONDITION": "Lighting",
            "ROADWAY_SURFACE_COND": "Road surface", "FIRST_CRASH_TYPE": "Crash type",
            "TRAFFICWAY_TYPE": "Road type", "PRIM_CONTRIBUTORY_CAUSE": "Cause"}


def options(col):
    vals = sorted(v for v in DF[col].dropna().unique())
    return [{"label": str(v).title().replace("/", " / "), "value": v} for v in vals]


def make_row(weather, light, surface, way, crashtype, cause, speed, units, hour, day, month) -> pd.DataFrame:
    return pd.DataFrame([{
        "WEATHER_CONDITION": weather,
        "LIGHTING_CONDITION": light,
        "ROADWAY_SURFACE_COND": surface,
        "TRAFFICWAY_TYPE": way,
        "FIRST_CRASH_TYPE": crashtype,
        "PRIM_CONTRIBUTORY_CAUSE": cause,
        "POSTED_SPEED_LIMIT": speed,
        "NUM_UNITS": units,
        "CRASH_HOUR": hour,
        "CRASH_DAY_OF_WEEK": day,
        "CRASH_MONTH": month,
    }])[NUMERIC + CATEGORICAL]


def nice_feature_name(raw: str) -> str:
    prefix, _, rest = raw.partition("__")
    if prefix == "num":
        return NICE_NUM.get(rest, rest.title())
    for col, nice in NICE_CAT.items():
        if rest.startswith(col + "_"):
            return f"{nice}: {rest[len(col) + 1:].title()}"
    return rest.title()


# ---------------------------------------------------------------------------
# Live figure: this crash vs. the city average
# ---------------------------------------------------------------------------
def fig_compare(p_injury: float, p_severe: float) -> go.Figure:
    labels = ["Someone<br>injured", "Serious<br>or fatal"]
    ratios = [p_injury / BASE_INJURY, p_severe / BASE_SEVERE]
    colors = [C["orange"] if r > 1 else C["blue"] for r in ratios]

    fig = go.Figure()
    fig.add_bar(x=labels, y=ratios, marker_color=colors,
                marker_line=dict(width=2, color=C["surface"]), width=0.5,
                text=[f"{r:.2f}x" for r in ratios], textposition="outside",
                textfont=dict(color=C["ink"], size=12),
                hovertemplate="%{x}<br>%{y:.2f}x the citywide average<extra></extra>")
    fig.update_layout(
        height=280, margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family=FONT, size=12, color=C["ink2"]),
        title=dict(text="This crash vs. the average Chicago crash",
                   font=dict(size=13, color=C["ink"]), x=0, xanchor="left"),
        showlegend=False,
        hoverlabel=dict(bgcolor=C["panel"], font_size=12, font_family=FONT,
                        bordercolor=C["border"], font_color=C["ink"]),
    )
    fig.add_hline(y=1.0, line=dict(color=C["ink3"], width=1, dash="dot"),
                 annotation_text="City's average", annotation_position="top left",
                 annotation_font=dict(size=10, color=C["ink3"]))
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(title="Multiple of the average crash",
                     gridcolor=C["grid"], zeroline=False, linecolor=C["border"],
                     rangemode="tozero", range=[0, max(1.35, max(ratios) * 1.3)])
    fig.update_traces(cliponaxis=False)
    return fig


CARD = {"background": C["panel"], "border": f"1px solid {C['border']}",
        "borderRadius": "10px", "padding": "20px"}
LABEL = {"fontSize": "11px", "fontWeight": 600, "letterSpacing": ".04em",
         "textTransform": "uppercase", "color": C["ink3"], "marginBottom": "5px",
         "display": "block"}


def control(label, component):
    return html.Div([html.Label(label, style=LABEL), component],
                    style={"marginBottom": "16px"})


# ---------------------------------------------------------------------------
# Page 1: Crash verdict
# ---------------------------------------------------------------------------
form = html.Div([
    html.Div("Describe the crash", style={"fontSize": "15px", "fontWeight": 600,
                                          "color": C["ink"], "marginBottom": "16px"}),
    control("Weather", dcc.Dropdown(id="weather", options=options("WEATHER_CONDITION"),
                                    value="CLEAR", clearable=False)),
    control("Lighting", dcc.Dropdown(id="light", options=options("LIGHTING_CONDITION"),
                                     value="DAYLIGHT", clearable=False)),
    control("Road surface", dcc.Dropdown(id="surface", options=options("ROADWAY_SURFACE_COND"),
                                         value="DRY", clearable=False)),
    control("Road type", dcc.Dropdown(id="way", options=options("TRAFFICWAY_TYPE"),
                                      value="NOT DIVIDED", clearable=False)),
    control("First crash type", dcc.Dropdown(id="crashtype", options=options("FIRST_CRASH_TYPE"),
                                              value="REAR END", clearable=False)),
    control("Primary contributory cause",
            dcc.Dropdown(id="cause", options=options("PRIM_CONTRIBUTORY_CAUSE"),
                         value=sorted(DF["PRIM_CONTRIBUTORY_CAUSE"].dropna().unique())[0],
                         clearable=False)),
    control("Posted speed limit (mph)",
            dcc.Slider(id="speed", min=15, max=55, step=5, value=30,
                       marks={v: {"label": str(v), "style": {"color": C["ink3"], "fontSize": "11px"}}
                              for v in range(15, 60, 10)},
                       tooltip={"placement": "bottom", "always_visible": False})),
    html.Div(style={"height": "10px"}),
    control("Number of units involved",
            dcc.Slider(id="units", min=1, max=6, step=1, value=2,
                       marks={v: {"label": str(v), "style": {"color": C["ink3"], "fontSize": "11px"}}
                              for v in range(1, 7)},
                       tooltip={"placement": "bottom", "always_visible": False})),
    html.Div(style={"height": "10px"}),
    control("Hour of day",
            dcc.Slider(id="hour", min=0, max=23, step=1, value=17,
                       marks={v: {"label": f"{v:02d}", "style": {"color": C["ink3"], "fontSize": "11px"}}
                              for v in [0, 6, 12, 18, 23]},
                       tooltip={"placement": "bottom", "always_visible": False})),
    html.Div(style={"height": "10px"}),
    control("Day of week", dcc.Dropdown(id="day", clearable=False, value=6,
                                        options=[{"label": v, "value": k} for k, v in DAYS.items()])),
    control("Month", dcc.Dropdown(id="month", clearable=False, value=8,
                                  options=[{"label": pd.Timestamp(2024, m, 1).strftime("%B"),
                                            "value": m} for m in range(1, 13)])),
], style={**CARD, "width": "340px", "flexShrink": 0, "alignSelf": "flex-start"})


def badge(icon: str, text: str, color: str):
    return html.Div([
        html.Span(icon, style={"display": "inline-flex", "alignItems": "center",
                               "justifyContent": "center", "width": "20px", "height": "20px",
                               "borderRadius": "50%", "background": "rgba(255,255,255,.28)",
                               "fontSize": "12px", "fontWeight": 800, "marginRight": "8px"}),
        html.Span(text),
    ], style={"display": "inline-flex", "alignItems": "center", "padding": "8px 16px 8px 10px",
             "borderRadius": "20px", "fontSize": "14px", "fontWeight": 700,
             "color": "#fff", "background": color})


def meter(value: float, base_rate: float, color: str):
    track_max = max(value * 1.15, base_rate * 5, 0.02)
    fill_pct = min(100.0, value / track_max * 100)
    marker_pct = min(96.0, base_rate / track_max * 100)

    return html.Div([
        html.Div([
            html.Div(style={"position": "absolute", "left": 0, "top": 0, "bottom": 0,
                            "width": f"{fill_pct}%", "borderRadius": "8px",
                            "background": color, "transition": "width .25s ease"}),
            html.Div(style={"position": "absolute", "left": f"{marker_pct}%", "top": "-3px",
                            "bottom": "-3px", "width": "2px", "background": C["ink"],
                            "opacity": 0.45}),
        ], style={"position": "relative", "height": "10px", "borderRadius": "8px",
                 "background": C["grid"], "overflow": "visible"}),
        html.Div([
            html.Span(f"citywide average {base_rate*100:.1f}%",
                     style={"position": "absolute", "left": f"{marker_pct}%",
                           "transform": "translateX(-50%)", "fontSize": "10.5px",
                           "color": C["ink3"], "whiteSpace": "nowrap"}),
        ], style={"position": "relative", "height": "16px", "marginTop": "4px"}),
    ])


def verdict_row(icon, text, color, prob_label, value, base_rate):
    return html.Div([
        html.Div([badge(icon, text, color),
                 html.Span(prob_label, style={"fontSize": "12.5px", "color": C["ink2"],
                                              "marginLeft": "10px"})],
                style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                      "gap": "4px", "marginBottom": "10px"}),
        meter(value, base_rate, color),
    ], style={"marginBottom": "20px"})


verdict = html.Div([
    html.Div("Verdict", style={"fontSize": "15px", "fontWeight": 600,
                               "color": C["ink"], "marginBottom": "16px"}),
    html.Div(id="verdict-injury"),
    html.Div(id="verdict-severity"),
], style=CARD)

verdict_charts = html.Div([
    html.Div(dcc.Graph(id="fig-compare", config={"displayModeBar": False}),
            style={**CARD, "flex": "1", "minWidth": 0}),
], style={"display": "flex", "gap": "12px", "marginTop": "12px"})

tab_verdict = html.Div([
    html.Div([
        form,
        html.Div([verdict, verdict_charts], style={"flex": "1", "minWidth": 0}),
    ], style={"display": "flex", "gap": "12px"}),
], style={"paddingTop": "18px"})


# ---------------------------------------------------------------------------
# Shared columns for the patterns & map pages
# ---------------------------------------------------------------------------
_dt = pd.to_datetime(DF["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
DF["CRASH_YEAR"] = _dt.dt.year
DF["CRASH_YEAR_MONTH"] = _dt.dt.to_period("M").astype(str)
DF["DAY_NAME"] = _dt.dt.day_name()
DF["FATAL"] = DF["MOST_SEVERE_INJURY"].eq("FATAL")
DF["SEVERITY_LABEL"] = "No injury"
DF.loc[DF["INJURY"] == 1, "SEVERITY_LABEL"] = "Injury"
DF.loc[DF["FATAL"], "SEVERITY_LABEL"] = "Fatal"

ALL_YEARS = sorted(int(y) for y in DF["CRASH_YEAR"].dropna().unique())

N_WEATHER_BUCKETS = 5
_weather_counts = DF["WEATHER_CONDITION"].value_counts()
_top_weather = _weather_counts.head(N_WEATHER_BUCKETS).index.tolist()
WEATHER_ORDER = [w.title() for w in _top_weather] + ["Other"]
DF["WEATHER_BUCKET"] = DF["WEATHER_CONDITION"].apply(lambda w: w.title() if w in _top_weather else "Other")

MIN_CRASH_TYPE_COUNT = 50
_ct_counts = DF["FIRST_CRASH_TYPE"].value_counts()
_rare_types = set(_ct_counts[_ct_counts < MIN_CRASH_TYPE_COUNT].index)
DF["CRASH_TYPE_BUCKET"] = DF["FIRST_CRASH_TYPE"].apply(
    lambda t: "Other" if t in _rare_types else t.title().replace(" To ", " to ")
)

_ym_sorted = sorted(DF["CRASH_YEAR_MONTH"].unique())
_monthly_all = DF.groupby("CRASH_YEAR_MONTH").size().reindex(_ym_sorted)
_is_low = {}
for _i, _ym in enumerate(_ym_sorted):
    if _i < 3:
        _is_low[_ym] = False
        continue
    _prior_avg = _monthly_all.iloc[_i - 3:_i].mean()
    _is_low[_ym] = bool(_monthly_all.iloc[_i] < 0.5 * _prior_avg)
INCOMPLETE_YM = []
for _ym in reversed(_ym_sorted):
    if _is_low[_ym]:
        INCOMPLETE_YM.append(_ym)
    else:
        break
INCOMPLETE_YM = sorted(INCOMPLETE_YM)

DATE_RANGE = (str(_dt.min().date()), str(_dt.max().date()))
TOTAL_ROWS = len(DF)

DF["LAT"] = pd.to_numeric(DF["LATITUDE"], errors="coerce")
DF["LON"] = pd.to_numeric(DF["LONGITUDE"], errors="coerce")
_has_geo = DF["LAT"].between(41.60, 42.10) & DF["LON"].between(-87.95, -87.50)
DF_GEO = DF.loc[_has_geo].copy()
GEO_SKIPPED = int(len(DF) - len(DF_GEO))


# ---------------------------------------------------------------------------
# Page 2: Crash patterns 
# ---------------------------------------------------------------------------
SEQ_BLUE = [
    [0.00, "#eef4fc"], [0.15, "#cde2fb"], [0.30, "#9ec5f4"],
    [0.45, "#6da7ec"], [0.60, "#3987e5"], [0.75, "#2a78d6"],
    [0.90, "#1c5cab"], [1.00, "#0d366b"],
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VIZ_LAYOUT = dict(plot_bgcolor=C["surface"], paper_bgcolor=C["surface"],
                  font=dict(color=C["ink"], family=FONT))
CHIP_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "6px"}


def viz_empty_figure(msg="No data for this selection"):
    fig = go.Figure()
    fig.update_layout(**VIZ_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=msg, showarrow=False, font=dict(color=C["ink3"]))])
    return fig


def viz_build_trend(d):
    if d.empty:
        return viz_empty_figure()
    monthly = d.groupby("CRASH_YEAR_MONTH").size().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, mode="lines", fill="tozeroy",
                             line=dict(color=C["blue"], width=2), fillcolor="rgba(42,120,214,0.12)",
                             hovertemplate="%{x}<br><b>%{y:,} crashes</b><extra></extra>", showlegend=False))
    fig.update_layout(**VIZ_LAYOUT, margin=dict(l=50, r=20, t=10, b=40),
                      xaxis=dict(showgrid=False, tickangle=-45, color=C["ink3"], tickfont=dict(size=10)),
                      yaxis=dict(title="Crashes", gridcolor=C["grid"], zeroline=False, color=C["ink3"]))
    return fig


def viz_build_hour_bar(d):
    if d.empty:
        return viz_empty_figure()
    by_hour = d.groupby("CRASH_HOUR").size().reindex(range(24), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_hour.index, y=by_hour.values, marker=dict(color=C["blue"]),
                         hovertemplate="Hour %{x}:00<br><b>%{y:,} crashes</b><extra></extra>", showlegend=False))
    fig.update_layout(**VIZ_LAYOUT, margin=dict(l=50, r=20, t=10, b=40),
                      xaxis=dict(title="Hour", dtick=2, showgrid=False, color=C["ink3"]),
                      yaxis=dict(title="Crashes", gridcolor=C["grid"], zeroline=False, color=C["ink3"]))
    return fig


def viz_build_weather_bar(d):
    if d.empty:
        return viz_empty_figure()
    counts = d["WEATHER_BUCKET"].value_counts().reindex(WEATHER_ORDER, fill_value=0)
    counts = counts.sort_values(ascending=True)
    total = counts.sum()
    pct = (counts / total * 100) if total else counts * 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts.values, y=counts.index, orientation="h", marker=dict(color=C["blue"]),
                         text=[f"{p:.0f}%" for p in pct], textposition="outside",
                         textfont=dict(color=C["ink3"], size=11), customdata=pct.values,
                         hovertemplate="<b>%{y}</b><br>%{x:,} crashes (%{customdata:.1f}% of selection)<extra></extra>",
                         showlegend=False))
    fig.update_layout(**VIZ_LAYOUT, margin=dict(l=100, r=40, t=10, b=40),
                      xaxis=dict(title="Crashes", gridcolor=C["grid"], zeroline=False, color=C["ink3"],
                                range=[0, counts.max() * 1.25 if counts.max() else 1]),
                      yaxis=dict(color=C["ink"], automargin=True))
    return fig


def viz_build_heatmap(d):
    if d.empty:
        return viz_empty_figure()
    pivot = d.pivot_table(index="DAY_NAME", columns="CRASH_HOUR", values="CRASH_YEAR",
                          aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=DAY_ORDER, fill_value=0).reindex(columns=range(24), fill_value=0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(range(24)), y=DAY_ORDER, colorscale=SEQ_BLUE,
        colorbar=dict(title="Crashes", outlinewidth=0, tickfont=dict(color=C["ink3"])),
        hovertemplate="%{y}, %{x}:00<br><b>%{z:,} crashes</b><extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_layout(**VIZ_LAYOUT, margin=dict(l=90, r=20, t=10, b=40),
                      xaxis=dict(title="Hour of day", dtick=2, color=C["ink3"]),
                      yaxis=dict(color=C["ink3"], autorange="reversed"))
    return fig


def viz_build_injury_chart(d):
    if d.empty:
        return viz_empty_figure()
    summary = d.groupby("CRASH_TYPE_BUCKET").agg(n=("CRASH_TYPE_BUCKET", "size"),
                                                  injured=("INJURY", "sum"), fatal=("FATAL", "sum"))
    summary = summary[summary["n"] > 0]
    if summary.empty:
        return viz_empty_figure()
    summary["rate"] = summary["injured"] / summary["n"] * 100
    summary = summary.sort_values("rate", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[summary["rate"].max() * 1.05] * len(summary), y=summary.index, orientation="h",
                         marker=dict(color=C["border"]), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(x=summary["rate"], y=summary.index, orientation="h", marker=dict(color=C["blue"]),
                         text=[f"{r:.1f}%" for r in summary["rate"]], textposition="outside",
                         textfont=dict(color=C["ink3"], size=12), customdata=summary[["n", "fatal"]].values,
                         hovertemplate=("<b>%{y}</b><br>%{x:.1f}% injury rate<br>"
                                        "%{customdata[0]:,} crashes &middot; %{customdata[1]:,} fatal<extra></extra>"),
                         showlegend=False))
    fig.update_layout(barmode="overlay", **VIZ_LAYOUT, margin=dict(l=170, r=60, t=10, b=40),
                      xaxis=dict(title="Share of crashes with a reported injury", ticksuffix="%",
                                showgrid=False, zeroline=False, color=C["ink3"],
                                range=[0, summary["rate"].max() * 1.18]),
                      yaxis=dict(showgrid=False, color=C["ink"], automargin=True),
                      height=90 + 32 * len(summary))
    return fig


def viz_control_block(label, children):
    return html.Div([html.Label(label, style=LABEL), *children])


def viz_chart_card(title, desc, graph_id):
    return html.Div(style={**CARD, "flex": "1 1 380px"}, children=[
        html.H2(title, style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P(desc, style={"margin": "0 0 6px", "fontSize": "12.5px", "color": C["ink3"]}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False}),
    ])


def viz_kpi_card(label, value):
    return html.Div(className="kpi-card", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ])


tab_patterns = html.Div([
    html.Div(style={**CARD, "display": "flex", "flexWrap": "wrap", "gap": "20px 32px", "marginBottom": "18px"},
            children=[
        viz_control_block("Years", [dcc.Checklist(id="viz-year-filter", className="chip-check",
            options=[{"label": str(y), "value": y} for y in ALL_YEARS], value=ALL_YEARS, style=CHIP_STYLE)]),
        viz_control_block("Weather", [dcc.Checklist(id="viz-weather-filter", className="chip-check",
            options=[{"label": w, "value": w} for w in WEATHER_ORDER], value=WEATHER_ORDER, style=CHIP_STYLE)])
    ]),
    html.Div(id="viz-kpi-row", style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "20px"}),
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}, children=[
        viz_chart_card("Crashes per month", "Monthly totals across the selected years", "viz-trend-chart"),
        viz_chart_card("Crashes by hour of day", "Summed across every selected year", "viz-hour-chart"),
        viz_chart_card("Crashes by weather condition",
                       "Clear weather dominates simply because most driving happens in clear weather "
                       "", "viz-weather-chart"),
    ]),
    html.Div(style={**CARD, "margin": "16px 0"}, children=[
        html.H2("Crashes by day & hour", style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P("Where day-of-week and time-of-day compound",
              style={"margin": "0 0 6px", "fontSize": "12.5px", "color": C["ink3"]}),
        dcc.Graph(id="viz-heat-chart", config={"displayModeBar": False}),
    ]),
    html.Div(style={**CARD, "margin": "16px 0"}, children=[
        html.H2("Injury rate by crash type", style={"margin": "0 0 2px", "fontSize": "15px", "fontWeight": 600}),
        html.P("Share of crashes with a reported injury, by first crash type",
              style={"margin": "0 0 6px", "fontSize": "12.5px", "color": C["ink3"]}),
        dcc.Graph(id="viz-injury-chart", config={"displayModeBar": False}),
    ]),
], style={"paddingTop": "18px"})


# ---------------------------------------------------------------------------
# Page 3: Crash map 
# ---------------------------------------------------------------------------

CAT_PALETTE = [C["blue"], C["orange"], "#1baf7a"]
OTHER_COLOR = "#b3b2ac"
SEVERITY_COLORS = {"No injury": C["good"], "Injury": C["warning"], "Fatal": C["critical"]}
SEVERITY_ORDER = ["No injury", "Injury", "Fatal"]

MAP_COLOR_OPTIONS = [
    {"label": "Crash severity", "value": "severity"},
    {"label": "Weather", "value": "WEATHER_CONDITION"},
    {"label": "Lighting", "value": "LIGHTING_CONDITION"},
    {"label": "Crash type", "value": "FIRST_CRASH_TYPE"},
    {"label": "Road layout", "value": "TRAFFICWAY_TYPE"},
    {"label": "Road surface", "value": "ROADWAY_SURFACE_COND"},
    {"label": "Primary cause", "value": "PRIM_CONTRIBUTORY_CAUSE"},
]
MAP_SAMPLE_CAP = 12000


def fig_map(selected_years, color_by):
    d = DF_GEO[DF_GEO["CRASH_YEAR"].isin(selected_years)]
    note = f"{len(d):,} crashes match these filters."
    if len(d) > MAP_SAMPLE_CAP:
        d = d.sample(MAP_SAMPLE_CAP, random_state=42)

    fig = go.Figure()
    if len(d):
        fig.add_trace(go.Densitymap(
            lat=d["LAT"], lon=d["LON"], radius=7, opacity=0.35, showscale=False,
            colorscale=[[0, "rgba(255,255,255,0)"], [1, C["ink"]]], hoverinfo="skip",
        ))

    if color_by == "severity":
        labels = d["SEVERITY_LABEL"]
        colors, groups = SEVERITY_COLORS, [g for g in SEVERITY_ORDER if g in set(labels)]
    else:
        vals = d[color_by].fillna("Missing").astype(str).str.title()
        top = vals.value_counts().head(3).index.tolist()
        labels = vals.where(vals.isin(top), "Other")
        groups = top + (["Other"] if "Other" in set(labels) else [])
        colors = {cat: CAT_PALETTE[i] for i, cat in enumerate(top)}
        colors["Other"] = OTHER_COLOR

    for g in groups:
        sub = d[labels == g]
        fig.add_trace(go.Scattermap(
            lat=sub["LAT"], lon=sub["LON"], mode="markers", name=g,
            marker=dict(size=6, color=colors[g]), opacity=0.75,
            hovertemplate=f"<b>{g}</b><extra></extra>",
        ))

    fig.update_layout(
        map=dict(style=MAP_STYLE, center=dict(lat=41.86, lon=-87.68), zoom=9.6),
        height=620, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor=C["border"], borderwidth=1,
                   font=dict(size=11, color=C["ink"]), x=0.01, y=0.99),
        font=dict(family=FONT, size=12),
        hoverlabel=dict(bgcolor=C["panel"], font_size=12, font_family=FONT,
                        bordercolor=C["border"], font_color=C["ink"]),
    )
    return fig, note


tab_map = html.Div([
    html.Div(style={**CARD, "display": "flex", "flexWrap": "wrap", "gap": "20px 32px",
                   "alignItems": "flex-end", "marginBottom": "18px"}, children=[
        viz_control_block("Years", [dcc.Checklist(id="map-year-filter", className="chip-check",
            options=[{"label": str(y), "value": y} for y in ALL_YEARS], value=ALL_YEARS, style=CHIP_STYLE)]),
        html.Div([
            html.Label("Color points by", style=LABEL),
            dcc.Dropdown(id="map-color-by", options=MAP_COLOR_OPTIONS, value="severity",
                        clearable=False, style={"width": "260px"}),
        ]),
    ]),
    html.Div(id="map-note", style={"fontSize": "11.5px", "color": C["ink3"], "marginBottom": "8px"}),
    html.Div(dcc.Graph(id="fig-map", config={"displayModeBar": False}), style={**CARD, "padding": "8px"}),
    html.Div(f"{GEO_SKIPPED:,} of {TOTAL_ROWS:,} crashes have no usable coordinates and are left off the map. ",
            style={"fontSize": "11.5px", "color": C["ink3"], "marginTop": "10px", "lineHeight": "1.5"}),
], style={"paddingTop": "18px"})


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Chicago Crash Dashboard")
server = app.server  # WSGI entry point for gunicorn (gunicorn app3:server)

_INDEX_CSS = f"""
        body {{ margin: 0; background: {C['surface']}; }}
        .chip-check input[type="checkbox"] {{ display: none; }}
        .chip-check label {{
            font-size: 12.5px; padding: 6px 12px; border-radius: 999px;
            border: 1px solid {C['border']}; background: transparent;
            color: {C['ink2']}; cursor: pointer; user-select: none;
            transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
        }}
        .chip-check label:hover {{ border-color: {C['blue']}; }}
        .chip-check input[type="checkbox"]:checked + label {{
            background: {C['blue']}; color: #ffffff; border-color: {C['blue']}; font-weight: 600;
        }}
        .lag-switch input[type="checkbox"] {{ transform: scale(1.15); margin-right: 8px; }}
        .kpi-card {{
            background: {C['panel']}; border: 1px solid {C['border']};
            border-radius: 10px; padding: 12px 18px; min-width: 150px; flex: 1 1 150px;
        }}
        .kpi-label {{ font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: {C['ink3']}; }}
        .kpi-value {{ font-size: 21px; font-weight: 700; margin-top: 4px; color: {C['ink']}; }}
"""
app.index_string = (
    "<!DOCTYPE html>\n<html>\n<head>\n"
    "    {%metas%}\n    <title>{%title%}</title>\n    {%favicon%}\n    {%css%}\n"
    "    <style>" + _INDEX_CSS + "    </style>\n"
    "</head>\n<body>\n    {%app_entry%}\n"
    "    <footer>\n        {%config%}\n        {%scripts%}\n        {%renderer%}\n    </footer>\n"
    "</body>\n</html>"
)

app.layout = html.Div([
    html.Div([
        html.H1("Chicago crash dashboard",
                style={"fontSize": "24px", "fontWeight": 700, "color": C["ink"],
                       "margin": "0 0 6px 0", "letterSpacing": "-.01em"}),
        html.P("Three views on the same 50,000-crash sample: get a per-crash severity "
               "verdict, explore citywide patterns, or see where crashes happen.",
               style={"fontSize": "13.5px", "color": C["ink2"], "margin": 0,
                      "maxWidth": "740px", "lineHeight": "1.55"}),
    ], style={"marginBottom": "18px"}),
    dcc.Tabs(id="tabs", value="verdict", children=[
        dcc.Tab(label="Crash verdict", value="verdict", children=tab_verdict),
        dcc.Tab(label="Crash patterns", value="patterns", children=tab_patterns),
        dcc.Tab(label="Crash map", value="map", children=tab_map),
    ], colors={"border": C["border"], "primary": C["blue"], "background": C["surface"]}),
], style={"fontFamily": FONT, "background": C["surface"], "minHeight": "100vh",
          "padding": "26px 30px 40px", "maxWidth": "1180px", "margin": "0 auto"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("verdict-injury", "children"),
    Output("verdict-severity", "children"),
    Output("fig-compare", "figure"),
    Input("weather", "value"), Input("light", "value"), Input("surface", "value"),
    Input("way", "value"), Input("crashtype", "value"), Input("cause", "value"),
    Input("speed", "value"), Input("units", "value"), Input("hour", "value"),
    Input("day", "value"), Input("month", "value"),
)
def update_verdict(weather, light, surface, way, crashtype, cause, speed, units, hour, day, month):
    row = make_row(weather, light, surface, way, crashtype, cause, speed, units, hour, day, month)
    p_injury = float(M_INJURY.predict_proba(row)[:, 1][0])
    p_severe = float(M_SEVERE.predict_proba(row)[:, 1][0])

    injury_yes = p_injury >= INJURY_THRESHOLD
    injury_block = verdict_row(
        "✓" if not injury_yes else "!", "NO INJURY" if not injury_yes else "INJURY",
        C["good"] if not injury_yes else C["warning"],
        f"Probability of someone getting injured = {p_injury*100:.1f}%", p_injury, BASE_INJURY,
    )

    if not injury_yes:
        severity_block = html.Div(
            "No severity call: the model does not expect an injury for these conditions.",
            style={"fontSize": "12.5px", "color": C["ink3"], "marginBottom": "8px"},
        )
    else:
        severe_yes = p_severe >= SEVERE_THRESHOLD
        severity_block = verdict_row(
            "!" if not severe_yes else "✕",
            "MINOR TO MODERATE" if not severe_yes else "SERIOUS OR FATAL",
            C["warning"] if not severe_yes else C["critical"],
            f"Probability of incapacitating or fatal injury = {p_severe*100:.2f}%", p_severe, BASE_SEVERE,
        )

    return injury_block, severity_block, fig_compare(p_injury, p_severe)


@app.callback(
    Output("viz-trend-chart", "figure"),
    Output("viz-hour-chart", "figure"),
    Output("viz-weather-chart", "figure"),
    Output("viz-heat-chart", "figure"),
    Output("viz-injury-chart", "figure"),
    Output("viz-kpi-row", "children"),
    Input("viz-year-filter", "value"),
    Input("viz-weather-filter", "value"),
)
def update_patterns(selected_years, selected_weather):
    selected_years = selected_years or ALL_YEARS
    selected_weather = selected_weather or WEATHER_ORDER

    d = DF[DF["CRASH_YEAR"].isin(selected_years) & DF["WEATHER_BUCKET"].isin(selected_weather)]

    n = len(d)
    if n == 0:
        empty = viz_empty_figure()
        return empty, empty, empty, empty, empty, [viz_kpi_card("Crashes in selection", "0")]

    n_months = d["CRASH_YEAR_MONTH"].nunique()
    avg_per_day = n / (n_months * 30.44) if n_months else 0
    busiest_day = d.groupby("DAY_NAME").size().reindex(DAY_ORDER, fill_value=0).idxmax()
    busiest_hour = int(d.groupby("CRASH_HOUR").size().idxmax())
    top_weather = d["WEATHER_BUCKET"].value_counts().idxmax()

    ct_summary = d.groupby("CRASH_TYPE_BUCKET").agg(n=("CRASH_TYPE_BUCKET", "size"), injured=("INJURY", "sum"))
    ct_summary = ct_summary[ct_summary["n"] >= 30]
    riskiest = (ct_summary["injured"] / ct_summary["n"]).idxmax() if not ct_summary.empty else "n/a"

    kpis = [
        viz_kpi_card("Crashes in selection", f"{n:,}"),
        viz_kpi_card("Avg crashes / day", f"{avg_per_day:.1f}"),
        viz_kpi_card("Busiest day", busiest_day),
        viz_kpi_card("Busiest hour", f"{busiest_hour:02d}:00"),
        viz_kpi_card("Top weather condition", top_weather),
        viz_kpi_card("Highest-risk crash type", riskiest),
    ]
    return (viz_build_trend(d), viz_build_hour_bar(d), viz_build_weather_bar(d),
            viz_build_heatmap(d), viz_build_injury_chart(d), kpis)


@app.callback(
    Output("fig-map", "figure"),
    Output("map-note", "children"),
    Input("map-year-filter", "value"),
    Input("map-color-by", "value"),
)
def update_map(selected_years, color_by):
    selected_years = selected_years or ALL_YEARS
    fig, note = fig_map(selected_years, color_by)
    return fig, note


if __name__ == "__main__":
    app.run(debug=False, port=8052)
