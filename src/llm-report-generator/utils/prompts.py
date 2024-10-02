## Collation of all prompts used for various sections of the report

system_prompt = """
You are working at an Australian financial institution and tasked to generate a report for end-of-day reporting.
The report includes business intelligence as well as regulatory compliance information which serves to provide clients with insights into their order activity for the day, 
and alert them to possible compliance issues.
"""

##TODO: Add example
order_summary_prompt = """
Generate a trade summary for the end-of-day report.
...
"""

anomaly_detection_prompt = system_prompt + """
You are tasked with writing the compliance section of the report. The orders have been run through an anomaly detection model.
Given the output, generate a report that highlights suspicious orders, and provide brief explanations for why they are suspicious.
You may reference the following rules that outline what constitutes a suspicious order:
a) ...
b) ...
c) ...
You may also reference some examples of suspicious orders and the explanations:
a) ...
b) ...
Please provide a report using the below output from the anomaly detection model:
... data...
"""