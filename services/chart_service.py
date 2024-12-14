# services/chart_service.py
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for chart generation
import matplotlib.pyplot as plt
from typing import List, Dict

class ChartService:
    def __init__(self):
        pass

    def generate_charts(self, charts_data: List[Dict]) -> List[str]:
        """
        charts_data: a list of dicts like:
        {
            "type": "bar" or "line" or "pie",
            "title": "Chart Title",
            "data": [{"label": "A", "value": 10}, {"label": "B", "value": 20}]
        }
        """
        chart_paths = []
        os.makedirs("charts", exist_ok=True)

        for i, chart in enumerate(charts_data):
            chart_type = chart["type"]
            title = chart["title"]
            data = chart["data"]

            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]

            plt.figure(figsize=(6,4))
            if chart_type == "bar":
                plt.bar(labels, values)
                plt.title(title)
                plt.xlabel("Category")
                plt.ylabel("Value")
            elif chart_type == "line":
                plt.plot(labels, values, marker='o')
                plt.title(title)
                plt.xlabel("Category")
                plt.ylabel("Value")
            elif chart_type == "pie":
                plt.pie(values, labels=labels, autopct='%1.1f%%')
                plt.title(title)
            else:
                # default to bar if unknown
                plt.bar(labels, values)
                plt.title(title)

            chart_path = f"charts/chart_{i}.png"
            plt.savefig(chart_path)
            plt.close()
            chart_paths.append(chart_path)

        return chart_paths
