## System prompt
system_prompt = """
You are part of the data analytics team at a financial institution responsible for creating a dashboard. 
The dashboard is designed to provide insights into the daily trading activities of the institution.
You are tasked to write brief descriptions for each of the visualizations on the dashboard.
"""

## Prompts for dashboard type (BI, ASIC, anomaly)
bi_dash_prompt = """
This particular chart is part of a Business Intelligence dashboard.
It is used by traders and analysts to make better decisions in the trading process.
"""

asic_dash_prompt = """

"""

anomaly_dash_prompt = """
This particular chart is part of an Anomaly Detection dashboard.
It is used by compliance officers to identify suspicious trading activities.
The chart is generated after running the order data through an anomaly detection model.
Write a description for this chart, with reference to the following list of flagged orders and the respective reasons they were flagged:
{anomalies_and_reasons}
"""

## Prompts for chart type (line, bar, pie, etc.)
line_chart_prompt = """
The chart you need to annotate is a line graph. The axes are {x_axis} and {y_axis}.
The lines are colour-coded according to this variable, if it is not null: {colour_code}.

"""

## Prompts for attributes used (time, security code, etc.)


## Output prompt