from datetime import date
from datetime import datetime
from datetime import time
import hashlib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import sys


@st.cache_data  # Cache the data loading so that every time the filter changes, data won't be loaded again
def load_data(file_path="data.xlsx"):
    df = pd.read_excel(file_path, sheet_name="data")
    df["Value"] *= df["ValueMultiplier"]
    df["Price"] *= df["PriceMultiplier"]

    def add_cumulative_columns(df, date_col, columns):
        df = df.sort_values(by=date_col, ascending=True)
        df.insert(0, f"{date_col}_CumulativeNumberOfOrders", range(1, len(df) + 1))
        for column in columns:
            df[f"{date_col}_Cumulative{column}"] = df[column].cumsum()
        return df

    columns = ["Quantity", "Value", "DoneVolume", "DoneValue"]
    df = add_cumulative_columns(df, "UpdateDate", columns)
    df = add_cumulative_columns(df, "CreateDate", columns)
    return df["AccCode"].unique(), df["Currency"].unique(), df["Exchange"].unique(), df


def filter_dataframe(df, column, options):
    return df[df[column].isin(options)]


# Load Data
list_of_traders, list_of_currencies, list_of_exchanges, df = load_data()

# Filter by Date
df = df[df["CreateDate"].dt.date == st.session_state.date]

# Sidebar Global Filters
enable_automated_report = st.sidebar.checkbox("Enable Automated Report?", value=True)
traders = st.sidebar.multiselect(
    label="Filter for Trader", options=list_of_traders, default=list_of_traders
)
df = filter_dataframe(df, "AccCode", traders)
exchanges = st.sidebar.multiselect(
    label="Filter for Exchange", options=list_of_exchanges, default=list_of_exchanges
)
df = filter_dataframe(df, "Exchange", exchanges)
currencies = st.sidebar.multiselect(
    label="Filter for Currency", options=list_of_currencies, default=list_of_currencies
)
df = filter_dataframe(df, "Currency", currencies)


# Dummy function for generating_automated_report
def generate_automated_report(fig):
    return "automated reported to be here"


def create_business_intelligence_dashboard():
    pass


def create_anomaly_detection_dashboard():
    st.subheader("Anomaly Detection Dashboard")
    container = st.container(border=True)
    container.write("Anomaly detection charts to be inserted")
    col1, col2, col3 = st.columns(3)
    # fig1 =
    # col1.plotly_chart(fig1)
    # ...
    # if (enable_automated_report):
    #   generate_automated_report(fig1)


def create_asic_reporting_dashboard():
    st.subheader("ASIC Reporting Dashboard")
    container = st.container(border=True)
    container.write("ASIC reporting charts to be inserted")

    # Dropdown options for bar chart and pie chart
    bar_chart_options = ["IntermediaryID", "OriginOfOrder"]
    pie_chart_options = ["OrderCapacity", "DirectedWholesale"]

    # Dropdowns for selecting the fields dynamically
    selected_bar_chart_field = st.selectbox(
        "Select field for Bar Chart:", bar_chart_options
    )
    selected_pie_chart_field = st.selectbox(
        "Select field for Pie Chart:", pie_chart_options
    )

    # Bar Chart based on the selected field
    st.subheader(f"Bar Chart - {selected_bar_chart_field}")
    bar_chart_data = df[selected_bar_chart_field].value_counts().reset_index()
    bar_chart_data.columns = [selected_bar_chart_field, "Count"]
    bar_fig = px.bar(
        bar_chart_data,
        x=selected_bar_chart_field,
        y="Count",
        title=f"Count of {selected_bar_chart_field}",
    )
    st.plotly_chart(bar_fig)

    # Pie Chart based on the selected field
    st.subheader(f"Pie Chart - {selected_pie_chart_field}")
    pie_chart_data = df[selected_pie_chart_field].value_counts().reset_index()
    pie_chart_data.columns = [selected_pie_chart_field, "Count"]
    pie_fig = px.pie(
        pie_chart_data,
        names=selected_pie_chart_field,
        values="Count",
        title=f"Count of {selected_pie_chart_field}",
    )
    st.plotly_chart(pie_fig)


if "Business Intelligence" in st.session_state.selections:
    create_business_intelligence_dashboard()

if "Anomaly Detection" in st.session_state.selections:
    create_anomaly_detection_dashboard()

if "ASIC Reporting" in st.session_state.selections:
    create_asic_reporting_dashboard()


# ============================================================
# Section: Metrics Container
# ============================================================
metrics_container = st.container(border=True)
(
    metrics_container_col1,
    metrics_container_col2,
    metrics_container_col3,
    metrics_container_col4,
    metrics_container_col5,
) = metrics_container.columns(5)
metrics_container_col1.metric("Total Number of Traders", len(df["AccCode"].unique()))
metrics_container_col2.metric(
    "Total Number of Distinct Secruity Codes Traded", len(df["SecCode"].unique())
)
metrics_container_col3.metric(
    "Total Number of Orders", f"{round((df.shape[0] + 1) / 1000, 2)}K"
)
metrics_container_col4.metric(
    "Total DoneVolume of Orders", f"{round(df['DoneVolume'].sum() / 1_000_000, 2)}M"
)
metrics_container_col5.metric(
    "Total DoneValue of Orders", f"{round(df['DoneValue'].sum()/ 1_000_000, 2)}M"
)


# ============================================================
# Section: Box Plot and Pie Chart
# ============================================================
def create_box_plot_for_each_security(
    parent_container,
    key,
    variable_options=[
        "Price",
        "DoneVolume",
        "DoneValue",
        "Quantity",
        "Value",
    ],
    variable_index=0,
):
    child_container = parent_container.container(border=True)
    variable = child_container.selectbox(
        label="Variable",
        options=variable_options,
        index=variable_index,
        key=key + "boxplot_variable",
        label_visibility="collapsed",
    )
    # Filter df for the top 5 security codes
    top5_sec_code = df["SecCode"].value_counts().nlargest(5).index
    fig = make_subplots(rows=1, cols=len(top5_sec_code))
    for i, sec in enumerate(top5_sec_code, start=1):
        fig.add_trace(
            go.Box(y=df[df["SecCode"] == sec][variable], name=sec), row=1, col=i
        )
    fig.update_layout(
        title=f"Box Plots of {variable} by Top 5 SecCode",
        height=180,
        margin=dict(t=40, b=0),
        showlegend=False,
    )
    child_container.plotly_chart(fig, key=key + "boxplot")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


def create_pie_chart_for_each_security(
    parent_container,
    key,
    variable_options=[
        "BuySell",
        "OrderSide",
        # "PriceInstruction",
        "Lifetime",
        "TimeInForce",
    ],
    variable_index=0,
):
    child_container = parent_container.container(border=True)
    variable = child_container.selectbox(
        label="Variable",
        options=variable_options,
        index=variable_index,
        key=key + "boxplot_variable",
        label_visibility="collapsed",
    )
    top5_sec_code = df["SecCode"].value_counts().nlargest(5).index
    specs = [[{"type": "domain"} for i in top5_sec_code]]
    fig = make_subplots(rows=1, cols=len(top5_sec_code), specs=specs)
    annotations = []
    for i, sec in enumerate(top5_sec_code, start=1):
        value_counts = df[df["SecCode"] == sec][variable].value_counts()
        fig.add_trace(
            go.Pie(labels=value_counts.index, values=value_counts, name=sec),
            row=1,
            col=i,
        )
        annotations.append(
            dict(
                text=sec,
                x=sum(fig.get_subplot(1, i).x) / 2,
                y=0.5,
                font_size=15,
                showarrow=False,
                xanchor="center",
            )
        )
    fig.update_traces(hole=0.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title=f"Pie Chart of {variable} by Top 5 SecCode",
        height=180,
        margin=dict(t=40, b=0),
        annotations=annotations,
    )
    child_container.plotly_chart(fig, key=key + "boxplot")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


first_row = st.container()
first_row_col1, first_row_col2 = first_row.columns([40, 60])
create_box_plot_for_each_security(parent_container=first_row_col1, key="4")
create_pie_chart_for_each_security(parent_container=first_row_col2, key="5")


# ============================================================
# Section: Box Plot and Pie Chart
# ============================================================
def create_pie_chart(parent_container, key):
    child_container = parent_container.container(border=True)
    variable = child_container.selectbox(
        label="Variable",
        options=[
            "Lifetime",
            # "PriceInstruction",
            "BuySell",
            "OrderSide",
            "Exchange",
            "Destination",
            "Currency",
            "OrderType",
            "TimeInForce",
        ],
        index=0,
        key=key + "pie_variable",
        label_visibility="collapsed",
    )
    value_counts = df[variable].value_counts()
    fig = px.pie(values=value_counts.values, names=value_counts.index).update_layout(
        title=f"Distribution of {variable}",
        height=200,
        margin=dict(t=30, b=0, l=0, r=0),
    )
    child_container.plotly_chart(fig, key=key + "pie")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


def create_bar_chart(parent_container, key):
    child_container = parent_container.container(border=True)
    variable = child_container.selectbox(
        label="Variable",
        options=[
            "PriceInstruction",
            "Lifetime",
            "BuySell",
            "OrderSide",
            "Exchange",
            "Destination",
            "Currency",
            "OrderType",
            "TimeInForce",
        ],
        index=0,
        key=key + "bar_var",
        label_visibility="collapsed",
    )
    value_counts = df[variable].value_counts()
    percentages = (value_counts / value_counts.sum() * 100).round(
        2
    )  # Calculate percentages
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={"y": "Count"},
        color_discrete_sequence=["peachpuff"],
    )
    for i, count in enumerate(value_counts.values):
        fig.add_annotation(
            x=value_counts.index[i],
            y=count,
            text=f"{percentages[i]}%",
            showarrow=False,
            font=dict(size=12),
            yshift=10,  # Adjusts the position of the text
        )
    fig.update_layout(
        title=f"Distribution of {variable}",
        height=200,
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis_title=None,
    )
    child_container.plotly_chart(fig, key=key + "bar")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


def create_histogram(
    parent_container,
    key,
    variable_options=[
        "DoneVolume",
        "DoneValue",
        "Quantity",
        "Value",
    ],
    variable_index=0,
):
    child_container = parent_container.container(border=True)
    variable = child_container.selectbox(
        label="Variable",
        options=variable_options,
        index=variable_index,
        key=key + "histogram_variable",
        label_visibility="collapsed",
    )
    fig = px.histogram(
        df, variable, color_discrete_sequence=["palegreen"]
    ).update_layout(
        title=f"Distribution of {variable}",
        height=200,
        margin=dict(t=30, b=0, l=0, r=0),
    )
    child_container.plotly_chart(fig, key=key + "histogram")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


second_row = st.container()
second_row_col1, second_row_col2, second_row_col3 = second_row.columns([20, 40, 40])
create_pie_chart(parent_container=second_row_col1, key="6")
create_bar_chart(parent_container=second_row_col2, key="7")
create_histogram(parent_container=second_row_col3, key="8")


# ============================================================
# Section: Line Chart and Sankey Diagram
# ============================================================
def create_line_chart(
    parent_container,
    key,
    x_axis_options=["CreateDate", "UpdateDate"],
    x_axis_index=0,
    y_axis_options=[
        "CumulativeNumberOfOrders",
        "CumulativeValue",
        "CumulativeDoneVolume",
        "CumulativeDoneValue",
        "CumulativeQuantity",
    ],
    y_axis_index=0,
):
    child_container = parent_container.container(border=True)
    col1, col2 = child_container.columns(2)
    y_axis = col1.selectbox(
        label="Y axis variable",
        options=y_axis_options,
        index=y_axis_index,
        key=key + "line_y_axis",
        label_visibility="collapsed",
    )
    x_axis = col2.selectbox(
        label="X axis variable",
        options=x_axis_options,
        index=x_axis_index,
        key=key + "line_x_axis",
        label_visibility="collapsed",
    )
    fig = px.line(df, x=x_axis, y=x_axis + "_" + y_axis)
    fig = fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
    child_container.plotly_chart(fig, key=key + "line")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


def create_sankey(
    parent_container,
    key,
    src_options=["SecCode", "Exchange"],
    src_index=0,
    target_options=["BuySell", "Destination"],
    target_index=0,
):
    child_container = parent_container.container(border=True)
    col1, col2 = child_container.columns(2)
    source = col1.selectbox(
        label="Source",
        options=src_options,
        index=src_index,
        key=key + "sankey_source",
        label_visibility="collapsed",
    )

    target = col2.selectbox(
        label="Target",
        options=target_options,
        index=target_index,
        key=key + "sankey_target",
        label_visibility="collapsed",
    )
    if source == "SecCode":
        top5_sec_code = df["SecCode"].value_counts().nlargest(5).index
        df_filtered = df[df["SecCode"].isin(top5_sec_code)]
    else:
        df_filtered = df
    agg_df = df_filtered.groupby([source, target]).size().reset_index(name="Count")
    # Prepare data for Sankey diagram - make target labels distinct by adding suffix
    source_labels = list(agg_df[source].unique())
    target_labels = [f"{label} (dest)" for label in df_filtered[target].unique()]
    labels = source_labels + target_labels
    # Create source index mapping
    source_index = {code: idx for idx, code in enumerate(source_labels)}
    # Initialize lists for Sankey data
    sources = []
    targets = []
    values = []
    # Fill source, target, and values
    for _, row in agg_df.iterrows():
        sources.append(source_index[row[source]])
        # Find the target index in the modified target labels
        target_label = f"{row[target]} (dest)"
        targets.append(labels.index(target_label))
        values.append(row["Count"])
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels
            ),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    fig = fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
    # fig.update_layout()
    child_container.plotly_chart(fig, key=key + "sankey")

    # Optional automated report section
    if "enable_automated_report" in globals() and enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


# Line Charts and Sankey Diagram
third_row = st.container()
third_row_col1, third_row_col2, third_row_col3 = third_row.columns(3)
create_line_chart(
    parent_container=third_row_col1,
    key="1",
)
create_line_chart(
    parent_container=third_row_col2,
    key="2",
    y_axis_index=1,
)
create_sankey(parent_container=third_row_col3, key="3")


# ============================================================
# Section: Letting users choose the chart type (Deprecated)
# ============================================================

# chart_type_1 = container1.selectbox(label="Chart type", options=["Line Chart", "Pie Chart"], index=0, label_visibility="collapsed")
# chart_type_2 = container2.selectbox(label="Chart type", options=["Line Chart", "Pie Chart"], index=1, label_visibility="collapsed")

# match chart_type_1:
#     case "Line Chart": create_line_chart(container1, "1")
#     case "Pie Chart": create_pie_chart(container1, "1")

# match chart_type_2:
#     case "Line Chart": create_line_chart(container2, "2")
#     case "Pie Chart": create_pie_chart(container2, "2")


# col1, col2, col3 = st.columns(3)  # Columns are containers
# # overview_container = st.container(border=True, key="overview_container")
# metrics_container = st.container(key="metrics_container")


# overview_container = col1.container(border=True)
# overview_expander = overview_container.expander("More Details")
# overview_expander.write("ZM insert your report here")


# with col1:


#     # with overview_container:
#     with metrics_container:
#         col_a, col_b, col_c = st.columns(3)
#         with col_a:
#             st.metric("Total Number of Orders", "5K")

#         with col_b:
#             st.metric("Total Volume of Orders", "40.39M")

#         with col_c:
#             st.metric("Total Value of Orders", "$33.28M")
#     with st.expander("More Details"):
#         st.write('''
#             ZM insert your report here.
#         ''')

# st.header("Selections")
# st.write(f"Your selections are {st.session_state.selections}.")

# def columns_array_creator(total_number, max_column_per_row):
#     columns_array = []
#     count = 0
#     while(count + max_column_per_row < total_number):
#         count += max_column_per_row
#         columns_array.append(max_column_per_row)
#     if(total_number - count > 0):
#         columns_array.append(total_number - count)
#     return columns_array

# selections = st.session_state.selections
# unique_values = set(item.split(" - ")[0] for item in selections)
# unique_count = len(unique_values)

# outer_columns_array = columns_array_creator(unique_count, 2)
# st.write(outer_columns_array)


# def create_sankey(
#     parent_container,
#     key,
#     src_options=["SecCode", "Exchange"],
#     src_index=0,
#     target_options=["BuySell", "Destination"],
#     target_index=0,
# ):
#     # Create child container
#     child_container = parent_container.container()

#     # Create selection boxes
#     source = child_container.selectbox(
#         label="Source",
#         options=src_options,
#         index=src_index,
#         key=key + "sankey_source",
#         label_visibility="collapsed",
#     )

#     target = child_container.selectbox(
#         label="Target",
#         options=target_options,
#         index=target_index,
#         key=key + "sankey_target",
#         label_visibility="collapsed",
#     )

#     # Filter data based on source selection
#     if source == "SecCode":
#         top5_sec_code = df["SecCode"].value_counts().nlargest(5).index
#         df_filtered = df[df["SecCode"].isin(top5_sec_code)]
#     else:
#         df_filtered = df

#     # Create aggregation based on selected source and target
#     agg_df = df_filtered.groupby([source, target]).size().reset_index(name="Count")

#     # Prepare data for Sankey diagram
#     labels = list(agg_df[source].unique()) + list(df_filtered[target].unique())

#     # Create source index mapping
#     source_index = {code: idx for idx, code in enumerate(agg_df[source].unique())}

#     # Initialize lists for Sankey data
#     sources = []
#     targets = []
#     values = []

#     # Fill source, target, and values
#     for _, row in agg_df.iterrows():
#         sources.append(source_index[row[source]])
#         targets.append(labels.index(row[target]))
#         values.append(row["Count"])

#     # Create Sankey diagram
#     fig = go.Figure(
#         go.Sankey(
#             node=dict(
#                 pad=15,
#                 thickness=20,
#                 line=dict(color="black", width=0.5),
#                 label=labels
#             ),
#             link=dict(
#                 source=sources,
#                 target=targets,
#                 value=values
#             )
#         )
#     )

#     # Display the chart
#     child_container.plotly_chart(fig, key=key + "sankey")

#     # Optional automated report section
#     if 'enable_automated_report' in globals() and enable_automated_report:
#         expander = child_container.expander("View automated report")
#         expander.write(generate_automated_report(fig))
