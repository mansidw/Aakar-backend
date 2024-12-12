import plotly.graph_objs as go
import plotly.io as pio
import os
from typing import List, Dict, Any

class ChartService:
    def __init__(self, charts_dir: str = "charts"):
        self.charts_dir = charts_dir
        os.makedirs(self.charts_dir, exist_ok=True)

    def generate_charts(self, charts_data: List[Dict[str, Any]]) -> List[str]:
        chart_paths = []
        for chart in charts_data:
            chart_type = chart.get('type')
            title = chart.get('title', 'Chart')
            data = chart.get('data', {})
            fig = None

            if chart_type.lower() == 'bar':
                fig = go.Figure([go.Bar(x=data.get('x', []), y=data.get('y', []))])
            elif chart_type.lower() == 'line':
                fig = go.Figure([go.Scatter(x=data.get('x', []), y=data.get('y', []), mode='lines')])
            elif chart_type.lower() == 'pie':
                fig = go.Figure([go.Pie(labels=data.get('labels', []), values=data.get('values', []))])
            else:
                continue

            if fig:
                fig.update_layout(title=title)
                chart_filename = f"{title.replace(' ', '_').lower()}.png"
                chart_path = os.path.join(self.charts_dir, chart_filename)
                try:
                    pio.write_image(fig, chart_path)
                    chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Failed to generate chart '{title}': {e}")
        return chart_paths
