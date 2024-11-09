## System prompt
system_prompt = """
You are part of the data analytics team at a financial institution responsible for creating a dashboard. 
The dashboard is designed to provide insights into the daily trading activities of the institution.
You are tasked to write brief descriptions for each of the visualizations on the dashboard,
covering how to interpret charts of this type, and drawing attention to important features (explicitly mention numerical values, if possible).
"""

## Prompts for dashboard type (BI, ASIC, anomaly)
bi_dash_prompt = """
This particular chart is part of a Business Intelligence dashboard.
A trader / analyst uses this dashboard to make informed decisions. Help them understand the chart.
What are some features they should pay attention to, and what insights can they gain from it?
Also provide one other avenue for further analysis that could be explored, which is not immediately obvious from the chart
(for example, comparing current data to historical data, or looking into possible reasons for certain trends).
Please keep your response short and concise, but make reference to numerical values where possible.
"""

##TODO: Add info on clauses, etc
asic_dash_prompt = """
This particular chart is part of a regulatory reporting dashboard. Users monitor the dashboard to ensure compliance with ASIC regulations.
Some charts are designed to highlight metrics that indicate compliance to specific clauses, while others are generic.
Here are some additional details about this particular chart:
{additional_info}
If a clause is specified, draw conclusions about the compliance status based on the chart, and explain your decision.
Otherwise, provide a general description of the chart and the insights that can be gained from it.
"""

anomaly_dash_prompt = """
This particular chart is part of an Anomaly Detection dashboard.
It is used by compliance officers to identify suspicious trading activities.
The chart is generated after running the order data through an anomaly detection model.
Write a description for this chart, with reference to the following list of flagged orders and the respective reasons they were flagged:
{anomalies_and_reasons}
"""


## Output prompt