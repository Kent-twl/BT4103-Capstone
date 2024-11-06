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
import anomaly

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(ROOT_DIR)
# from src.llm.utils.fig_description_generator import fig_description_generator
#functional import
from src.functions.asic_functions import *

# @st.cache_data
# def anom_results():
#     anomaly_df = anomaly.load_excel('data.xlsx')
#     continuous_outlier_df, discrete_outlier_df = anomaly.outlier_results(anomaly_df.copy())
#     anomalies = anomaly.anomaly_results(anomaly_df.copy())
#     return anomaly_df, continuous_outlier_df, discrete_outlier_df, anomalies

# anomaly_df, continuous_outlier_df, discrete_outlier_df, anomalies = anom_results()


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
    df["ExecutionVenueCategory"] = df["ExecutionVenue"].apply(classify_execution_venue)
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


def generate_description(fig, dash_type="bi", chart_type="line", vars=[]):
    # Generate text in the LLM based on fig
    ret_output = fig_description_generator(fig, dash_type, chart_type, vars)

    # Return the text
    return ret_output


# Dummy function for generating_automated_report
def generate_automated_report(fig):
    return "automated reported to be here"


def create_business_intelligence_dashboard():
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
            "ExecutionTime",
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
                "ExecutionTime",
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
                "ExecutionTime",
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
            "Value",
            "DoneVolume",
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


def create_anomaly_detection_dashboard():
    st.header("Anomaly Detection Dashboard")
    with st.container(border=True):
        st.subheader("Outlier Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Continuous Outliers")
            st.divider()
            if not continuous_outlier_df.empty:
                st.dataframe(continuous_outlier_df, height = 300)
            else:
                st.write("There are no continuous outliers identified.")
        with col2:
            st.write("Discrete Outliers")
            st.divider()
            if not discrete_outlier_df.empty:
                st.dataframe(discrete_outlier_df, height = 300)
            else:
                st.write("There are no discrete outliers identified.")
    with st.container(border=True):
        st.subheader("Anomaly Detection Results")
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
            
            # Calculate ExecutionTime in minutes
            filtered_df['ExecutionTime'] = round((filtered_df['UpdateDate'] - filtered_df['CreateDate']).dt.total_seconds() / 60)
            filtered_df['CumulativeNumberOfOrders'] = filtered_df['CreateDate_CumulativeNumberOfOrders']

            # Calculate the count and percentage of orders filled immediately (ExecutionTime = 0)
            immediate_filled_orders = filtered_df[filtered_df['ExecutionTime'] == 0]['CumulativeNumberOfOrders'].sum()
            total_orders = filtered_df['CumulativeNumberOfOrders'].sum()
            immediate_filled_percentage = (immediate_filled_orders / total_orders) * 100 if total_orders > 0 else 0

            # Display the metric with both the percentage and raw count
            st.markdown("""
                <div style="text-align: center; padding: 10px; border-radius: 8px; background-color: #000000; color: white;">
                    <h3>Orders Filled Immediately</h3>
                    <h1 style="font-size: 40px;">{:.2f}%</h1>
                    <p style="font-size: 18px;">({:,} out of {:,} orders)</p>
                </div>
            """.format(immediate_filled_percentage, immediate_filled_orders, total_orders), unsafe_allow_html=True)

            # Filter data for orders not filled immediately (ExecutionTime > 0)
            not_filled_immediately_df = filtered_df[filtered_df['ExecutionTime'] > 0]

            # Group by ExecutionTime and AccCode for the line graph
            area_data = not_filled_immediately_df.groupby(['ExecutionTime', 'AccCode'])['CumulativeNumberOfOrders'].sum().reset_index()

            # Create the line graph for orders not filled immediately
            area_fig = px.area(
                area_data, 
                x="ExecutionTime", 
                y="CumulativeNumberOfOrders", 
                color="AccCode",
                title=f"{selected_chart} - {selected_acccode if selected_acccode != 'All' else 'All Accounts'}",
                labels={"ExecutionTime": "Execution Time (minutes)", "CumulativeNumberOfOrders": "Cumulative Number of Orders"}
            )

            # Display the line graph in Streamlit
            st.plotly_chart(area_fig)
            
            # Show additional details if a specific AccCode is selected
            if selected_acccode != "All":
                # Display cumulative orders over time for the selected AccCode
                cumulative_orders = filtered_df[['CreateDate', 'CumulativeNumberOfOrders']].drop_duplicates().sort_values(by='CreateDate')
                cumulative_orders_fig = px.bar(cumulative_orders, x="CreateDate", y="CumulativeNumberOfOrders",
                                                title="Cumulative Number of Orders Over Time",
                                                labels={"CreateDate": "Timestamp", "CumulativeNumberOfOrders": " Number of Orders"})
                st.plotly_chart(cumulative_orders_fig)
                
                # # Calculate the count and percentage of orders filled immediately
                # immediate_fill_count = filtered_df[filtered_df['ExecutionTime'] == 0]['CumulativeNumberOfOrders'].sum()
                # total_order_count = filtered_df['CumulativeNumberOfOrders'].sum()
                # immediate_fill_percentage = (immediate_fill_count / total_order_count) * 100 if total_order_count > 0 else 0

                # # Display the metrics with custom styling
                # st.markdown("""
                #     <div style="text-align: center; padding: 10px; border-radius: 8px; background-color: #4CAF50; color: white;">
                #         <h3>Orders Filled Immediately</h3>
                #         <h1 style="font-size: 40px;">{:.2f}%</h1>
                #         <p style="font-size: 18px;">({:,} out of {:,} orders)</p>
                #     </div>
                # """.format(immediate_fill_percentage, immediate_fill_count, total_order_count), unsafe_allow_html=True)

                # # Display a breakdown of orders by ExecutionTime within this specific AccCode
                # execution_time_data = filtered_df.groupby('ExecutionTime')['CumulativeNumberOfOrders'].sum().reset_index()
                # execution_time_fig = px.bar(
                #     execution_time_data, 
                #     x="ExecutionTime", 
                #     y="CumulativeNumberOfOrders",
                #     title="Number of Orders by Time Taken for Execution",
                #     labels={"ExecutionTime": "Execution Time", "CumulativeNumberOfOrders": "Number of Orders"}
                # )
                # st.plotly_chart(execution_time_fig)

                # Show additional data details in a table
                st.write("Additional data details:")
                st.dataframe(filtered_df[['AccCode', 'CreateDate', 'UpdateDate', 'OrderNo', 'OrderSide', 'OrderType', 'ExecutionTime', 'DoneVolume', 'Price', 'Quantity', 'ExecutionVenue']])
            
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
                'DoneVolume': 'sum',  # sum DoneValue
                'Value': 'sum'  # sum Value
            }).reset_index()
            summary_data.columns = ['AccCode', 'Total Done Volume', 'Total Cumulative Value']
            st.dataframe(summary_data)
            
            # Show additional details if a specific AccCode is selected
            if selected_acccode != "All":
                st.markdown(f"### Detailed View for Account Code: {selected_acccode}")
                
                # Display cumulative done volume over time for the selected AccCode
                cumulative_done_volume = filtered_df[['CreateDate', 'DoneVolume']].drop_duplicates().sort_values(by='CreateDate')
                done_volume_fig = px.line(cumulative_done_volume, x="CreateDate", y="DoneVolume",
                                        title="Done Volume Over Time",
                                        labels={"CreateDate": "Date", "DoneVolume": "Done Volume"})
                st.plotly_chart(done_volume_fig)
                
                # Display cumulative value over time for the selected AccCode
                cumulative_value = filtered_df[['CreateDate', 'Value']].drop_duplicates().sort_values(by='CreateDate')
                value_fig = px.line(cumulative_value, x="CreateDate", y="Value",
                                    title="Order Value Over Time",
                                    labels={"CreateDate": "Date", "Value": "Order Value"})
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
