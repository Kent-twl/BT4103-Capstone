import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data  # Cache the data loading so that every time the filter changes, data won't be loaded again
def load_data(file_path="data.xlsx"):
    df = pd.read_excel(file_path, sheet_name="data")
    df.insert(0, "CumulativeNumberOfOrders", range(1, len(df) + 1))
    df["Value"] = df["Value"] * df["ValueMultiplier"]
    df["Price"] = df["Price"] * df["PriceMultiplier"]
    df["CumulativeValue"] = df["Value"].cumsum()
    df["CumulativeDoneVolume"] = df["DoneVolume"].cumsum()
    df["OriginOfOrder"] = df["OriginOfOrder"].astype(str)
    return df

df = load_data()

enable_automated_report = st.sidebar.checkbox("Enable Automated Report?", value=True)

def generate_automated_report(fig):
    return "Zhe Ming add in your text here / call your function here"

def create_business_intelligence_dashboard():
    pass

def create_anomaly_detection_dashboard():
    st.subheader("Anomaly Detection Dashboard")
    container = st.container(border=True)
    container.write("Anomaly detection charts to be inserted")
    col1, col2, col3 = st.columns(3)

def create_asic_reporting_dashboard():
    st.title("ASIC Reporting Dashboard")
    
    st.write("Please select the relevant fields you would like to see for ASIC Reporting.")

    # Dropdown options for the charts
    pie_chart_options = ['OrderCapacity', 'DirectedWholesale']
    bar_chart_options = ['IntermediaryID', 'OriginOfOrder', 'OrderTakerUserCode', 'OrderGiver','OrderCapacity', 'DirectedWholesale']
    
    # Multi-select for pie chart fields with no default selection
    st.markdown("#### Select fields for Pie Charts:")
    st.markdown("**Note:** Only categorical fields with less than 4 categories can be selected for pie charts.")
    selected_pie_fields = st.multiselect("Choose fields for pie chart", pie_chart_options, default=[])

    # Multi-select for bar chart fields with no default selection
    st.markdown("#### Select fields for Bar Charts:")
    st.markdown("**Note:** Only categorical fields can be selected for bar charts.")
    selected_bar_fields = st.multiselect("Choose fields for bar chart", bar_chart_options, default=[])

    # Option to choose bar chart orientation
    bar_chart_orientation = st.selectbox("Select bar chart orientation:", ["", "Vertical", "Horizontal"])

    # Generate pie charts for the selected fields
    if selected_pie_fields:
        for field in selected_pie_fields:
            st.subheader(f"{field}")
            pie_chart_data = df[field].value_counts().reset_index()
            pie_chart_data.columns = [field, 'Count']
            
            # Ensure data types are compatible
            pie_chart_data['Count'] = pie_chart_data['Count'].astype(int)
            
            pie_fig = px.pie(pie_chart_data, names=field, values='Count', title=f"Count of {field}")
            col1, col2 = st.columns([2, 1])  # Create two columns: one for the chart and one for the table

            with col1:
                st.plotly_chart(pie_fig)

            with col2:
                st.write("Summary Table:")
                st.dataframe(pie_chart_data)

    # Generate bar charts for the selected fields
# Generate bar charts for the selected fields
    if selected_bar_fields:
        for field in selected_bar_fields:
            st.subheader(f"{field}")

            # Create the bar chart data
            bar_chart_data = df[field].value_counts().reset_index()
            bar_chart_data.columns = [field, 'Count']

            # Ensure all expected categories are included
            all_categories = df[field].unique()
            for category in all_categories:
                if category not in bar_chart_data[field].values:
                    bar_chart_data = bar_chart_data.append({field: category, 'Count': 0}, ignore_index=True)

            # Ensure data types are compatible
            bar_chart_data['Count'] = bar_chart_data['Count'].astype(int)
            bar_chart_data[field] = bar_chart_data[field].astype(str)

            # Create the bar chart based on selected orientation
            if bar_chart_orientation == "Vertical":
                bar_fig = px.bar(bar_chart_data, x=field, y='Count', title=f"Count of {field}")
            elif bar_chart_orientation == "Horizontal":
                bar_fig = px.bar(bar_chart_data, y=field, x='Count', title=f"Count of {field}", orientation='h')
            else:
                st.warning("Please select a bar chart orientation.")
                continue  # Skip plotting if no valid orientation is selected

            # Display the chart and summary table
            col1, col2 = st.columns([2, 1])  # Create two columns: one for the chart and one for the table

            with col1:
                st.plotly_chart(bar_fig)

            with col2:
                st.write("Summary Table:")
                st.dataframe(bar_chart_data)


# Example call to the function (you can uncomment this for testing)
# create_asic_reporting_dashboard()

if "Business Intelligence" in st.session_state.selections:
    create_business_intelligence_dashboard()

if "Anomaly Detection" in st.session_state.selections:
    create_anomaly_detection_dashboard()

if "ASIC Reporting" in st.session_state.selections:
    create_asic_reporting_dashboard()

def create_line_chart(
    parent_container,
    key,
    x_axis_options=["CreateDate", "DeleteDate"],
    x_axis_index=0,
    y_axis_options=["CumulativeNumberOfOrders", "Value", "Quantity"],
    y_axis_index=0,
):
    child_container = parent_container.container()
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
        options=["SecCode"],
        index=0,
        key=key + "sankey_source",
        label_visibility="collapsed",
    )
    target = child_container.selectbox(
        label="Target",
        options=["BuySell"],
        index=0,
        key=key + "sankey_target",
        label_visibility="collapsed",
    )
    agg_df = df.groupby(['Exchange', 'BuySell']).size().reset_index(name='Count')

    # Prepare data for Sankey diagram
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

    # Create the Sankey diagram
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
