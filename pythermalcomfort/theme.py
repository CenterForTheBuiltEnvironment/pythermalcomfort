import plotly.graph_objects as go
import plotly.io as pio

custom_template = pio.templates["plotly_white"].update(
    layout_plot_bgcolor="rgba(255,255,255,1)",
    layout_paper_bgcolor="rgba(255,255,255,1)",
    layout_font_family="Arial",
    layout_font_color="black",
)

pio.templates["pythermalcomfort_template"] = custom_template

pio.templates.default = "pythermalcomfort_template"

# mm = 1/(2.54*10)
# journal_column_widths = {"single_column": 89*mm, "double_column": 183*mm}

tight_margins = dict(l=35, r=35, t=45, b=20)

index_mapping_dictionary = {
    "HI": {
        "name": "Heat Index",
        "order_categories": [
            "No Risk",
            "Caution",
            "Extreme<br>Caution",
            "Danger",
            "Extreme<br>Danger",
        ],
        "colors": ["#FFFF65", "#FFD602", "#FF8C00", "#FF0000"],
        "colors_categories": ["grey", "#FFFF65", "#FFD602", "#FF8C00", "#FF0000"],
        "si": {
            "unit": "°C",
            "range": [27, 60],
        },
        "ip": {
            "unit": "°F",
            "range": [81, 140],
        },
        "conversion_function": "temperature",
    },
}
