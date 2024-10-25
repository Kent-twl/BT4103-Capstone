import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
import os
import plotly.graph_objects as go



@st.cache_data  # Cache the data loading so that everytime the filter changes data won't be loaded again
def load_data(file_path="data.xlsx"):
    df = pd.read_excel(file_path, sheet_name="data")
    df.insert(0, "CumulativeNumberOfOrders", range(1, len(df) + 1))
    df["Value"] = df["Value"] * df["ValueMultiplier"]
    df["Price"] = df["Price"] * df["PriceMultiplier"]
    df["CumulativeValue"] = df["Value"].cumsum()
    df["CumulativeDoneVolume"] = df["DoneVolume"].cumsum()
    return df
df = load_data()

enable_automated_report = st.sidebar.checkbox("Enable Automated Report?", value=True)
# Add global level filters for sidebar?


def generate_automated_report(fig):
    # Generate text in the LLM based on fig
    # Return the text
    return "Zhe Ming add in your text here / call your function here"


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


if "Business Intelligence" in st.session_state.selections:
    create_business_intelligence_dashboard()

if "Anomaly Detection" in st.session_state.selections:
    create_anomaly_detection_dashboard()

if "ASIC Reporting" in st.session_state.selections:
    create_asic_reporting_dashboard()


## I'll clean up my charts in a bit.


def create_line_chart(
    parent_container,
    key,
    x_axis_options=["CreateDate", "DeleteDate"],
    x_axis_index=0,
    y_axis_options=["CumulativeNumberOfOrders", "Value", "Quantity"],
    y_axis_index=0,
):
    child_container = parent_container.container()
    # col1, col2 = child_container.columns(2)
    y_axis = child_container.selectbox(
        label="Y axis variable",
        options=y_axis_options,
        index=y_axis_index,
        key=key + "line_y_axis",
        label_visibility="collapsed",
    )
    x_axis = child_container.selectbox(
        label="X axis variable",
        options=x_axis_options,
        index=x_axis_index,
        key=key + "line_x_axis",
        label_visibility="collapsed",
    )
    fig = px.line(df, x=x_axis, y=y_axis).update_layout(height=250)
    child_container.plotly_chart(fig, key=key + "line")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))


def create_pie_chart(parent_container, key):
    child_container = parent_container.container()
    variable = child_container.selectbox(
        label="Variable",
        options=["OrderCapacity", "Exchange"],
        index=0,
        key=key + "pie_var",
        label_visibility="collapsed",
    )
    fig = px.pie(df, values="Price", names=variable).update_layout(height=250)
    child_container.plotly_chart(fig, key=key + "pie")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))

def create_sankey(parent_container, key):
    child_container = parent_container.container()
    source = child_container.selectbox(
        label="Source",
        # options=y_axis_options,
        # index=y_axis_index,
        options=["SecCode"],
        index=0,
        key=key + "sankey_source",
        label_visibility="collapsed",
    )
    target = child_container.selectbox(
        label="Target",
        # options=x_axis_options,
        # index=x_axis_index,
        options=["BuySell"],
        index=0,
        key=key + "sankey_target",
        label_visibility="collapsed",
    )
    agg_df = df.groupby(['Exchange', 'BuySell']).size().reset_index(name='Count')

    # Step 3: Prepare data for Sankey diagram
    labels = agg_df['Exchange'].unique().tolist() + ['B', 'S']
    source = []
    target = []
    values = []

    # Map SecCode to their respective indices
    sec_code_index = {code: idx for idx, code in enumerate(agg_df['Exchange'].unique())}

    # Fill source, target, and values
    for _, row in agg_df.iterrows():
        source.append(sec_code_index[row['Exchange']])  # Index of SecCode
        target.append(labels.index(row['BuySell']))     # Index of BuySell
        values.append(row['Count'])

    # Step 4: Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        )
    ))
    child_container.plotly_chart(fig, key=key + "pie")
    if enable_automated_report:
        expander = child_container.expander("View automated report")
        expander.write(generate_automated_report(fig))

# Create the BI Dashboard
st.subheader("Business Intelligence Dashboard")
metrics_container = st.container(border=True)
metrics_container_col1, metrics_container_col2, metrics_container_col3, metrics_container_col4, metrics_container_col5 = metrics_container.columns(5)
metrics_container_col1.metric("Total Number of Traders", len(df["AccCode"].unique()))
metrics_container_col2.metric("Number of Secruity Codes Traded", len(df["SecCode"].unique()))
metrics_container_col3.metric("Total Number of Orders", f"{round((df.shape[0] + 1) / 1000, 2)}K")
metrics_container_col4.metric("Total Volume of Orders", f"{round(df['DoneVolume'].sum() / 1_000_000, 2)}M")
metrics_container_col5.metric("Total Value of Orders", f"{round(df['Value'].sum()/ 1_000_000, 2)}M")

st_col1, st_col2 = st.columns(spec=[2, 1], vertical_alignment="top")
col1_container = st_col1.container(border=True)
col2_container = st_col2.container(border=True)
col1_container_col1, col1_container_col2 = col1_container.columns(2)
create_line_chart(
    parent_container=col1_container_col1,
    key="1",
    x_axis_options=["CreateDate", "DeleteDate"],
    y_axis_options=[
        "CumulativeNumberOfOrders",
        "CumulativeValue",
        "CumulativeDoneVolume",
    ],
)
create_line_chart(
    parent_container=col1_container_col2,
    key="2",
    x_axis_options=["CreateDate", "DeleteDate"],
    y_axis_options=[
        "CumulativeNumberOfOrders",
        "CumulativeValue",
        "CumulativeDoneVolume",
    ],
    y_axis_index=1,
)
create_sankey(
    parent_container=col2_container, 
    key="3"
)

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
