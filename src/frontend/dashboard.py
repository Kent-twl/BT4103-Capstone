import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
import os
import sys
import plotly.graph_objects as go
import anomaly

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(ROOT_DIR)
# from src.llm.utils.fig_description_generator import fig_description_generator
#functional import
from src.functions.asic_functions import *

@st.cache_data
def anom_results():
    anomaly_df = anomaly.load_excel('data.xlsx')
    continuous_outlier_df, discrete_outlier_df = anomaly.outlier_results(anomaly_df.copy())
    anomalies = anomaly.anomaly_results(anomaly_df.copy())
    return anomaly_df, continuous_outlier_df, discrete_outlier_df, anomalies

anomaly_df, continuous_outlier_df, discrete_outlier_df, anomalies = anom_results()

@st.cache_data  # Cache the data loading so that every time the filter changes, data won't be loaded again
def load_data(file_path="data.xlsx"):
    df = pd.read_excel(file_path)
    df.insert(0, "CumulativeNumberOfOrders", range(1, len(df) + 1))
    df["Value"] = df["Value"] * df["ValueMultiplier"]
    df["Price"] = df["Price"] * df["PriceMultiplier"]
    df["CumulativeValue"] = df["Value"].cumsum()
    df["CumulativeDoneVolume"] = df["DoneVolume"].cumsum()
    df["OriginOfOrder"] = df["OriginOfOrder"].astype(str)
    #Threshold for Execution Venue Classification
    df["ExecutionVenueCategory"] = df["ExecutionVenue"].apply(classify_execution_venue)
    return df

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
        expander.write(generate_description(fig))

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
        expander.write(generate_description(fig))

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
        expander.write(generate_description(fig))

df = load_data()

enable_automated_report = st.sidebar.checkbox("Enable Automated Report?", value=True)

def generate_description(fig, dash_type="bi", chart_type="line", vars=[]):
    # Generate text in the LLM based on fig
    ret_output = fig_description_generator(fig, dash_type, chart_type, vars)

    # Return the text
    return ret_output

def create_business_intelligence_dashboard():
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

def create_anomaly_detection_dashboard():
    st.header("Anomaly Detection Dashboard")
    with st.container(border=True):
        st.subheader("Outlier Analysis Results")
        st.write("Continuous Outliers")
        st.dataframe(continuous_outlier_df, height = 300)
        st.divider()
        st.write("Discrete Outliers")
        st.dataframe(discrete_outlier_df, height = 300)
    with st.container(border=True):
        st.subheader("Anomaly Detection Results")
        st.write("Scatterplot of Price vs Quantity, with Anomalies marked out")  
        st.write("Please select the field you would like to see the data coloured by.")
        scatter_options = ['SecCode', 'AccCode', 'OrderSide', 'OrderGiver', 'OriginOfOrder', 'Exchange', 'Destination']
        selected_field = st.selectbox('Field: ', scatter_options)
        if selected_field:
            fig = anomaly.show_overall_scatterplot(anomaly_df, anomalies, selected_field)
            st.pyplot(fig)
        else:
            fig = anomaly.show_overall_scatterplot(anomaly_df, anomalies, 'SecCode')
            st.pyplot(fig)
        st.divider()
        st.write("Anomalous Trades")
        st.table(anomalies)

def create_asic_reporting_dashboard():
    #User Selection for Dashboard Type
    st.title("ASIC Reporting Dashboard")
    chart_type = st.selectbox("Choose Chart type", options=["User-specified Dashboard", "Clause-specific Dashboard"])

    #Workflow for User-tailored Dashboard
    if chart_type == "User-specified Dashboard":
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
    
    #Clause-specific workflow           
    elif chart_type == 'Clause-specific Dashboard':
        # Dropdown menu for clause-specific charts
        asic_chart_options = [
            "RG 265.130 (Best Execution Obligation)",
            "RG 265.12 (Supervision of Market Participants)",
            "RG 265.51 (Suspicious Activity Reporting)",
            "RG 265.74 (Crossing Systems Reporting)"
        ]
        selected_chart = st.selectbox("Select the Reporting Standard you want to view:", asic_chart_options)
        
        # Display selected chart
        if selected_chart == "RG 265.130 (Best Execution Obligation)":
            st.subheader(selected_chart)
            
            # Dropdown to select a specific AccCode for a detailed view
            unique_acccodes = df["AccCode"].unique()
            selected_acccode = st.selectbox("Select an Account Code for a detailed view", options=["All"] + list(unique_acccodes))
            
            # Filter data based on the selected AccCode
            if selected_acccode != "All":
                filtered_df = df[df["AccCode"] == selected_acccode]
            else:
                filtered_df = df
            
            # Group by TimeInForce and AccCode to show cumulative number of orders
            area_data = filtered_df.groupby(['TimeInForce', 'AccCode'])['CumulativeNumberOfOrders'].sum().reset_index()
            area_fig = px.area(area_data, x="TimeInForce", y="CumulativeNumberOfOrders", color="AccCode",
                            title=f"{selected_chart} - {selected_acccode if selected_acccode != 'All' else 'All Accounts'}",
                            labels={"TimeInForce": "Time In Force", "CumulativeNumberOfOrders": "Cumulative Number of Orders"})
            st.plotly_chart(area_fig)
            
            # Show additional details if a specific AccCode is selected
            if selected_acccode != "All":
                st.markdown(f"### Detailed View for Account Code: {selected_acccode}")
                
                # Display cumulative orders over time for the selected AccCode
                cumulative_orders = filtered_df[['CreateDate', 'CumulativeNumberOfOrders']].drop_duplicates().sort_values(by='CreateDate')
                cumulative_orders_fig = px.line(cumulative_orders, x="CreateDate", y="CumulativeNumberOfOrders",
                                                title="Cumulative Number of Orders Over Time",
                                                labels={"CreateDate": "Date", "CumulativeNumberOfOrders": "Cumulative Number of Orders"})
                st.plotly_chart(cumulative_orders_fig)
                
                # Display a breakdown of orders by TimeInForce within this specific AccCode
                time_in_force_data = filtered_df.groupby('TimeInForce')['CumulativeNumberOfOrders'].sum().reset_index()
                time_in_force_fig = px.bar(time_in_force_data, x="TimeInForce", y="CumulativeNumberOfOrders",
                                        title="Number of Orders by Time Taken for Execution",
                                        labels={"TimeInForce": "Time In Force", "CumulativeNumberOfOrders": "Number of Orders"})
                st.plotly_chart(time_in_force_fig)
                
                # Show additional data details in a table
                st.write("Additional data details:")
                st.dataframe(filtered_df[['AccCode','CreateDate', 'OrderNo', 'OrderSide', 'OrderType', 'TimeInForce',  'DoneVolume', 'Price', 'Quantity', 'ExecutionVenue']])
            
        elif selected_chart == "RG 265.12 (Supervision of Market Participants)":
            st.subheader(selected_chart)
            
            # Dropdown to select a specific AccCode
            unique_acccodes = df["AccCode"].unique()
            selected_acccode = st.selectbox("Select an Account Code for a detailed view", options=["All"] + list(unique_acccodes))
            
            # Filter data based on the selected AccCode
            if selected_acccode != "All":
                filtered_df = df[df["AccCode"] == selected_acccode]
            else:
                filtered_df = df
            
            # Summary data for all AccCodes or filtered by selected AccCode
            summary_data = filtered_df.groupby('AccCode').agg({
                'CumulativeDoneVolume': 'last',  # Assuming cumulative sum is already in `CumulativeDoneVolume`
                'CumulativeValue': 'last'  # Assuming cumulative sum is in `CumulativeValue`
            }).reset_index()
            summary_data.columns = ['AccCode', 'Total Done Volume', 'Total Cumulative Value']
            st.dataframe(summary_data)
            
            # Show additional details if a specific AccCode is selected
            if selected_acccode != "All":
                st.markdown(f"### Detailed View for Account Code: {selected_acccode}")
                
                # Display cumulative done volume over time for the selected AccCode
                cumulative_done_volume = filtered_df[['CreateDate', 'CumulativeDoneVolume']].drop_duplicates().sort_values(by='CreateDate')
                done_volume_fig = px.line(cumulative_done_volume, x="CreateDate", y="CumulativeDoneVolume",
                                        title="Cumulative Done Volume Over Time",
                                        labels={"CreateDate": "Date", "CumulativeDoneVolume": "Cumulative Done Volume"})
                st.plotly_chart(done_volume_fig)
                
                # Display cumulative value over time for the selected AccCode
                cumulative_value = filtered_df[['CreateDate', 'CumulativeValue']].drop_duplicates().sort_values(by='CreateDate')
                value_fig = px.line(cumulative_value, x="CreateDate", y="CumulativeValue",
                                    title="Cumulative Order Value Over Time",
                                    labels={"CreateDate": "Date", "CumulativeValue": "Cumulative Value"})
                st.plotly_chart(value_fig)
                
                # Display a breakdown of average done volume per order type within this specific AccCode
                avg_done_volume_data = filtered_df.groupby('OrderType')['DoneVolume'].mean().reset_index()
                avg_done_volume_fig = px.bar(avg_done_volume_data, x="OrderType", y="DoneVolume",
                                            title="Average Done Volume by Order Type",
                                            labels={"OrderType": "Order Type", "DoneVolume": "Average Done Volume"})
                st.plotly_chart(avg_done_volume_fig)
                
                # Show additional data details in a table
                st.write("Additional data details:")
                st.dataframe(filtered_df[['AccCode','CreateDate','OrderSide', 'OrderType', 'DoneVolume', 'Price', 'Quantity', 'ExecutionVenue']])

        elif selected_chart == "RG 265.51 (Suspicious Activity Reporting)":
            # User input for large order threshold
            threshold = st.number_input("Threshold for large orders (number of orders):", min_value=1, value=10)
            
            # Classify orders based on the threshold
            df_with_classification = classify_orders_based_on_threshold(df.copy(), threshold)
            
            # Dropdown to select a specific AccCode for detailed view
            unique_acccodes = df_with_classification["AccCode"].unique()
            selected_acccode = st.selectbox("Select an Account Code for a detailed view", options=["All"] + list(unique_acccodes))
            
            # Filter data based on the selected AccCode
            if selected_acccode != "All":
                filtered_df = df_with_classification[df_with_classification["AccCode"] == selected_acccode]
            else:
                filtered_df = df_with_classification
            
            # Group by the new classification and AccCode for the bar chart
            bar_data = filtered_df.groupby(['OrderSizeCategory', 'AccCode'])['DoneVolume'].sum().reset_index()
            bar_fig = px.bar(bar_data, x="AccCode", y="DoneVolume", color="OrderSizeCategory", 
                            barmode='group', title=f"RG 265.51 (Suspicious Activity Reporting) - {selected_acccode if selected_acccode != 'All' else 'All Accounts'}",
                            labels={"DoneVolume": "Total Done Volume", "AccCode": "Account Code", "OrderSizeCategory": "Order Size"})
            st.plotly_chart(bar_fig)
            
            # Show additional details if a specific AccCode is selected
            if selected_acccode != "All":
                st.markdown(f"### Detailed View for Account Code: {selected_acccode}")
                
                # Display a breakdown of order types within this specific account code
                order_type_data = filtered_df.groupby(['OrderType', 'OrderSizeCategory']).size().reset_index(name='Count')
                order_type_fig = px.bar(order_type_data, x="OrderType", y="Count", color="OrderSizeCategory", 
                                        barmode='stack', title="Order Types within Selected Account Code",
                                        labels={"OrderType": "Order Type", "Count": "Number of Orders", "OrderSizeCategory": "Order Size Category"})
                st.plotly_chart(order_type_fig)
                
                # Display total value of orders within this specific account code
                value_data = filtered_df.groupby('OrderSizeCategory')['Value'].sum().reset_index()
                value_fig = px.bar(value_data, x="OrderSizeCategory", y="Value", 
                                title="Total Order Value within Selected Account Code",
                                labels={"OrderSizeCategory": "Order Size Category", "Value": "Total Order Value"})
                st.plotly_chart(value_fig)
                
                # Show a table for additional data if needed
                st.write("Additional data details:")
                st.dataframe(filtered_df[['AccCode','OrderNo','OrderSide','OrderType', 'DoneVolume', 'Price', 'Quantity', 'Value', 'ExecutionVenue']])

        elif selected_chart == "RG 265.74 (Crossing Systems Reporting)":
            st.subheader(selected_chart)
            # Dropdown to select a specific exchange system
            unique_exchange_systems = df["ExecutionVenueCategory"].unique()
            selected_exchange_system = st.selectbox("Select an Exchange System for a detailed view", options=["All"] + list(unique_exchange_systems))
            
            # Filter data based on the selected exchange system
            if selected_exchange_system != "All":
                filtered_df = df[df["ExecutionVenueCategory"] == selected_exchange_system]
            else:
                filtered_df = df
            
            # Treemap of all exchange systems or filtered by selected system
            tree_data = filtered_df.groupby(['ExecutionVenueCategory', 'AccCode']).size().reset_index(name='Count of Orders')
            tree_fig = px.treemap(tree_data, path=['ExecutionVenueCategory', 'AccCode'], values='Count of Orders', 
                                title=f"{selected_chart} - {selected_exchange_system if selected_exchange_system != 'All' else 'All Systems'}",
                                labels={"ExecutionVenueCategory": "Execution Venue", "AccCode": "Account Code", "Count of Orders": "Number of Orders"})
            st.plotly_chart(tree_fig)
            
            # Show additional details if a specific system is selected
            if selected_exchange_system != "All":
                st.markdown(f"### Detailed View for {selected_exchange_system}")
                
                # Display a breakdown of order types within this specific exchange system
                order_type_data = filtered_df.groupby(['OrderType', 'AccCode']).size().reset_index(name='Count')
                order_type_fig = px.bar(order_type_data, x="AccCode", y="Count", color="OrderType", 
                                        barmode='stack', title="Order Types within Selected Exchange System",
                                        labels={"AccCode": "Account Code", "Count": "Number of Orders", "OrderType": "Order Type"})
                st.plotly_chart(order_type_fig)
                
                # Display total done volume within this specific exchange system
                volume_data = filtered_df.groupby('AccCode')['DoneVolume'].sum().reset_index()
                volume_fig = px.bar(volume_data, x="AccCode", y="DoneVolume", 
                                    title="Total Done Volume within Selected Exchange System",
                                    labels={"AccCode": "Account Code", "DoneVolume": "Done Volume"})
                st.plotly_chart(volume_fig)
                
                # Show a table for additional data if needed
                st.write("Additional data details:")
                st.dataframe(filtered_df[['AccCode', 'OrderNo','OrderSide', 'OrderType', 'DoneVolume', 'Price', 'Quantity', 'ExecutionVenue']])

# Call to the function (you can uncomment this for testing)
if "Business Intelligence" in st.session_state.selections:
    create_business_intelligence_dashboard()

if "Anomaly Detection" in st.session_state.selections:
    create_anomaly_detection_dashboard()

if "ASIC Reporting" in st.session_state.selections:
    create_asic_reporting_dashboard()