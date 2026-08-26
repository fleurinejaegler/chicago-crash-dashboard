
# %%
import calendar
import colorsys
import html
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


DATA_FILE = Path(__file__).with_name("chicago_crashes_recent_sample_50000.csv")

REQUIRED_COLUMNS = {"CRASH_DATE", "LATITUDE", "LONGITUDE"}

# These columns represent quantities, so the dashboard gives them a range slider.
# Every other selectable column is treated as a category and gets a multi-select.
NUMERIC_COLUMNS = {
    "POSTED_SPEED_LIMIT",
    "NUM_UNITS",
    "INJURIES_TOTAL",
    "CRASH_HOUR",
}

FILTER_COLUMNS = [
    "POSTED_SPEED_LIMIT",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "FIRST_CRASH_TYPE",
    "TRAFFICWAY_TYPE",
    "ROADWAY_SURFACE_COND",
    "CRASH_TYPE",
    "PRIM_CONTRIBUTORY_CAUSE",
    "NUM_UNITS",
    "CRASH_MONTH",
    "MOST_SEVERE_INJURY",
    "INJURIES_TOTAL",
    "INJURIES_FATAL",
    "CRASH_DAY_OF_WEEK",
]

# Friendly names shown in the dashboard. The CSV column names remain unchanged.
COLUMN_LABELS = {
    "None": "No additional filter (color by crash severity)",
    "POSTED_SPEED_LIMIT": "Posted speed limit",
    "WEATHER_CONDITION": "Weather",
    "LIGHTING_CONDITION": "Lighting conditions",
    "FIRST_CRASH_TYPE": "Type of collision",
    "TRAFFICWAY_TYPE": "Road layout",
    "ROADWAY_SURFACE_COND": "Road surface condition",
    "CRASH_TYPE": "Crash outcome",
    "PRIM_CONTRIBUTORY_CAUSE": "Primary cause",
    "NUM_UNITS": "Number of vehicles involved",
    "CRASH_MONTH": "Month of crash",
    "MOST_SEVERE_INJURY": "Most severe injury",
    "INJURIES_TOTAL": "Total injuries",
    "INJURIES_FATAL": "Fatalities",
    "CRASH_HOUR": "Hour of day",
    "CRASH_DAY_OF_WEEK": "Day of week",
}

NUMERIC_PALETTE = [
    [49, 130, 189, 170],
    [107, 174, 214, 170],
    [253, 174, 97, 170],
    [244, 109, 67, 170],
    [215, 48, 39, 170],
]


def column_label(column: str) -> str:
    """Return a readable interface label for a dataset column."""
    return COLUMN_LABELS.get(column, column.replace("_", " ").title())


def categorical_values(data: pd.DataFrame, column: str) -> pd.Series:
    """Return readable grouping values for categorical dashboard filters."""
    if column == "CRASH_MONTH":
        month_numbers = pd.to_numeric(data[column], errors="coerce")
        values = month_numbers.map(
            lambda value: calendar.month_name[int(value)]
            if pd.notna(value) and 1 <= int(value) <= 12
            else "Missing"
        )
        return values.astype(str)

    if column == "CRASH_DAY_OF_WEEK":
        # Chicago's crash dataset uses 1 = Sunday and 7 = Saturday.
        day_numbers = pd.to_numeric(data[column], errors="coerce")
        values = pd.Series("Missing", index=data.index, dtype="object")
        values.loc[day_numbers.isin([1, 7])] = "Weekend"
        values.loc[day_numbers.isin([2, 3, 4, 5, 6])] = "Weekday"
        return values

    if column == "INJURIES_FATAL":
        fatality_numbers = pd.to_numeric(data[column], errors="coerce")
        values = pd.Series("Missing", index=data.index, dtype="object")
        values.loc[fatality_numbers.eq(0)] = "0 fatalities"
        values.loc[fatality_numbers.gt(0)] = "1+ fatalities"
        return values

    return data[column].fillna("Missing").astype(str)


def ordered_categories(values: pd.Series, column: str) -> list[str]:
    """Put calendar and binary groups in a natural order."""
    present = set(values.unique().tolist())
    if column == "CRASH_MONTH":
        preferred = list(calendar.month_name)[1:] + ["Missing"]
    elif column == "CRASH_DAY_OF_WEEK":
        preferred = ["Weekday", "Weekend", "Missing"]
    elif column == "INJURIES_FATAL":
        preferred = ["0 fatalities", "1+ fatalities", "Missing"]
    else:
        return sorted(present)
    return [category for category in preferred if category in present]


@st.cache_data
def load_data(path: Path) -> tuple[pd.DataFrame, int]:
    """Load the crash data and return the mappable rows and skipped-row count."""
    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The CSV is missing required columns: {missing}")

    data["CRASH_DATE_PARSED"] = pd.to_datetime(
        data["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    data["YEAR"] = data["CRASH_DATE_PARSED"].dt.year.astype("Int64")
    data["LATITUDE"] = pd.to_numeric(data["LATITUDE"], errors="coerce")
    data["LONGITUDE"] = pd.to_numeric(data["LONGITUDE"], errors="coerce")

    for column in NUMERIC_COLUMNS.intersection(data.columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    has_coordinates = (
        data["LATITUDE"].notna()
        & data["LONGITUDE"].notna()
        & data["LATITUDE"].ne(0)
        & data["LONGITUDE"].ne(0)
    )
    skipped_rows = int((~has_coordinates).sum())
    return data.loc[has_coordinates].copy(), skipped_rows


def categorical_filter(
    data: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, bool]:
    """Draw a category selector and return its filtered data."""
    display_values = categorical_values(data, column)
    categories = ordered_categories(display_values, column)

    include_all = st.sidebar.checkbox(
        "Include all groups", value=True, key=f"all_{column}"
    )
    if include_all:
        chosen = categories
        st.sidebar.caption(f"All {len(categories)} groups are included.")
    else:
        chosen = st.sidebar.multiselect(
            "Groups to display",
            options=categories,
            default=categories[:1],
            key=f"groups_{column}",
        )

    return data.loc[display_values.isin(chosen)].copy(), include_all


def numeric_filter(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Draw a range slider and return its filtered data."""
    values = data[column].dropna()
    if values.empty:
        st.sidebar.warning(f"{column} has no numeric values for these years.")
        return data.iloc[0:0].copy()

    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        st.sidebar.info(f"All values are {minimum:g}.")
        return data.loc[data[column].eq(minimum)].copy()

    integers_only = bool((values % 1 == 0).all())
    if integers_only:
        selected_range = st.sidebar.slider(
            "Range to display",
            min_value=int(minimum),
            max_value=int(maximum),
            value=(int(minimum), int(maximum)),
            key=f"range_{column}",
        )
    else:
        selected_range = st.sidebar.slider(
            "Range to display",
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
            key=f"range_{column}",
        )

    return data.loc[data[column].between(*selected_range, inclusive="both")].copy()


def categorical_color(index: int, total: int) -> list[int]:
    """Return visually distinct colors for an arbitrary number of groups."""
    hue = index / max(total, 1)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.88)
    return [int(red * 255), int(green * 255), int(blue * 255), 155]


def add_point_colors(
    data: pd.DataFrame, selected_column: str
) -> tuple[pd.DataFrame, str, list[tuple[str, list[int]]]]:
    """Color points by severity, category, or binned numeric values."""
    data = data.copy()

    if selected_column == "None":
        fatal_values = data["INJURIES_FATAL"].fillna(0)
        injury_values = data["INJURIES_TOTAL"].fillna(0)
        colors = {
            "No injury": [32, 120, 210, 145],
            "Injury": [245, 145, 40, 160],
            "Fatality": [210, 35, 45, 185],
        }
        data["POINT_COLOR"] = [
            colors["Fatality"]
            if fatal > 0
            else colors["Injury"]
            if injured > 0
            else colors["No injury"]
            for fatal, injured in zip(fatal_values, injury_values)
        ]
        return data, "Crash severity", list(colors.items())

    if selected_column not in NUMERIC_COLUMNS:
        values = categorical_values(data, selected_column)
        groups = ordered_categories(values, selected_column)
        if selected_column == "INJURIES_FATAL":
            fatality_colors = {
                "0 fatalities": [32, 120, 210, 165],
                "1+ fatalities": [210, 35, 45, 195],
                "Missing": [125, 125, 125, 150],
            }
            color_map = {group: fatality_colors[group] for group in groups}
        else:
            color_map = {
                group: categorical_color(index, len(groups))
                for index, group in enumerate(groups)
            }
        data["POINT_COLOR"] = values.map(color_map)
        entries = [(group, color_map[group]) for group in groups]
        return data, column_label(selected_column), entries

    values = data[selected_column]
    unique_count = int(values.nunique())
    if unique_count <= 1:
        color = NUMERIC_PALETTE[0]
        data["POINT_COLOR"] = [color] * len(data)
        label = "Missing" if values.dropna().empty else f"{values.dropna().iloc[0]:g}"
        return data, column_label(selected_column), [(label, color)]

    bin_count = min(5, unique_count)
    minimum = float(values.min())
    maximum = float(values.max())
    edges = [
        minimum + (maximum - minimum) * index / bin_count
        for index in range(bin_count + 1)
    ]
    codes = pd.cut(
        values,
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    colors = NUMERIC_PALETTE[: len(edges) - 1]
    data["POINT_COLOR"] = [
        [125, 125, 125, 145] if pd.isna(code) else colors[int(code)]
        for code in codes
    ]
    entries = [
        (f"{edges[index]:g} – {edges[index + 1]:g}", colors[index])
        for index in range(len(colors))
    ]
    return data, column_label(selected_column), entries


def show_map_legend(title: str, entries: list[tuple[str, list[int]]]) -> None:
    """Show a compact legend immediately above the map's top-left corner."""
    items = "".join(
        "<span style='display:inline-flex;align-items:center;margin:3px 12px 3px 0;'>"
        f"<span style='width:11px;height:11px;border-radius:50%;margin-right:5px;"
        f"background:rgb({color[0]},{color[1]},{color[2]});'></span>"
        f"{html.escape(str(label))}</span>"
        for label, color in entries
    )
    st.markdown(
        "<div style='display:inline-block;max-width:100%;max-height:135px;"
        "overflow-y:auto;background:white;border:1px solid #d8dee4;border-radius:6px;"
        "padding:7px 10px;margin-bottom:5px;font-size:12px;color:#222;'>"
        f"<div><b>Point color: {html.escape(title)}</b></div>"
        f"<div style='display:flex;flex-wrap:wrap;'>{items}</div>"
        "<div style='margin-top:3px;border-top:1px solid #eee;padding-top:4px;'>"
        "<b>Crash concentration (gray background):</b>"
        "<span style='display:inline-block;width:115px;height:9px;margin:0 6px;"
        "border:1px solid #aaa;background:linear-gradient(90deg,#fff,#111);'></span>"
        "Fewer&nbsp;&nbsp;→&nbsp;&nbsp;More</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def make_tooltip(columns: list[str]) -> dict[str, str]:
    visible = [
        column
        for column in columns
        if column not in {"CRASH_DATE_PARSED", "POINT_COLOR"}
    ]
    midpoint = (len(visible) + 1) // 2
    left, right = visible[:midpoint], visible[midpoint:]
    rows = []
    for index, left_column in enumerate(left):
        cells = [left_column]
        if index < len(right):
            cells.append(right[index])
        rows.append(
            "<tr>"
            + "".join(
                "<td style='padding:3px 10px 3px 0;vertical-align:top;"
                "max-width:280px;overflow-wrap:anywhere;'>"
                f"<b>{column.replace('_', ' ').title()}:</b> {{{column}}}</td>"
                for column in cells
            )
            + "</tr>"
        )
    tooltip_html = (
        "<div style='font-size:11px;line-height:1.25;'>"
        "<table style='border-collapse:collapse;'>"
        + "".join(rows)
        + "</table></div>"
    )
    return {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "#17202a",
            "maxWidth": "620px",
            "maxHeight": "80vh",
            "overflowY": "auto",
            "whiteSpace": "normal",
        },
    }


st.set_page_config(page_title="Chicago Traffic Crashes", layout="wide")
st.title("Chicago Traffic Crash Map")
st.caption(
    "Hover over a point to see its crash record. Use the heat layer to identify "
    "areas with a higher concentration of accidents."
)

if not DATA_FILE.exists():
    st.error(f"Could not find the dataset at {DATA_FILE}")
    st.stop()

try:
    crashes, skipped = load_data(DATA_FILE)
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

st.sidebar.header("Filters")
available_years = sorted(crashes["YEAR"].dropna().astype(int).unique().tolist())
selected_years = st.sidebar.multiselect(
    "Crash year",
    options=available_years,
    default=available_years,
    help="Choose one or more years.",
)
filtered = crashes.loc[crashes["YEAR"].isin(selected_years)].copy()

available_filters = [column for column in FILTER_COLUMNS if column in crashes.columns]
selected_column = st.sidebar.selectbox(
    "Filter and color points by",
    options=["None"] + available_filters,
    format_func=column_label,
)

all_groups_selected = False
if selected_column != "None":
    if selected_column in NUMERIC_COLUMNS:
        filtered = numeric_filter(filtered, selected_column)
    else:
        filtered, all_groups_selected = categorical_filter(filtered, selected_column)

st.sidebar.divider()
st.sidebar.metric("Points on map", f"{len(filtered):,}")
st.sidebar.caption(
    f"{skipped:,} source rows cannot be mapped because coordinates are missing or zero."
)

if selected_column != "None" and selected_column not in NUMERIC_COLUMNS:
    st.sidebar.subheader(f"Count by {column_label(selected_column)}")
    group_counts = (
        categorical_values(filtered, selected_column)
        .value_counts()
        .rename_axis("Group")
        .reset_index(name="Points")
    )
    st.sidebar.dataframe(group_counts, hide_index=True, width="stretch")
    if all_groups_selected:
        st.sidebar.caption("Counts include every group in the selected years.")

if filtered.empty:
    st.warning("No crashes match the selected filters.")
    st.stop()

filtered, legend_title, legend_entries = add_point_colors(filtered, selected_column)
view_state = pdk.ViewState(
    latitude=float(filtered["LATITUDE"].mean()),
    longitude=float(filtered["LONGITUDE"].mean()),
    zoom=10.2,
    pitch=0,
)
heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=filtered,
    get_position="[LONGITUDE, LATITUDE]",
    get_weight=1,
    radius_pixels=45,
    intensity=1,
    threshold=0.04,
    opacity=0.48,
    color_range=[
        [255, 255, 255, 0],
        [225, 225, 225, 80],
        [175, 175, 175, 115],
        [115, 115, 115, 145],
        [60, 60, 60, 180],
        [15, 15, 15, 210],
    ],
)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position="[LONGITUDE, LATITUDE]",
    get_fill_color="POINT_COLOR",
    get_radius=35,
    radius_min_pixels=3,
    radius_max_pixels=10,
    opacity=0.72,
    stroked=True,
    get_line_color=[255, 255, 255, 190],
    line_width_min_pixels=0.6,
    pickable=True,
    auto_highlight=True,
)
deck = pdk.Deck(
    layers=[heat_layer, point_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip=make_tooltip(list(crashes.columns)),
)

# A centered portrait panel better matches Chicago's north-south geography.
left_space, map_column, right_space = st.columns([1, 2, 1])
with map_column:
    show_map_legend(legend_title, legend_entries)
    st.pydeck_chart(deck, width="stretch", height=820)
