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
    "adaptive": {
        "name": "Adaptive Comfort Model",
        "order_categories": [
            "Out of Range",
            "Acceptable<br>for 90%",
            "Acceptable<br>for 80%",
            "Discomfort",
        ],
        "colors_categories": [
            "grey",
            "rgba(0, 128, 0, 0.2)",
            "rgba(0, 0, 255, 0.2)",
            "rgba(255, 0, 0, 0.2)",
        ],
        "si": {
            "unit": "°C",
            "range": [10, 33.5],
        },
        "ip": {
            "unit": "°F",
            "range": [50, 92.5],
        },
        "conversion_function": "temperature",
    },
    "pmv": {
        "name": "Predicted Mean Vote",
        "order_categories": [
            "Out of Range",
            "Too Cold",
            "Comfortable",
            "Too Hot",
        ],
        "colors_categories": [
            "grey",
            "rgba(0, 0, 255, 0.2)",
            "rgba(0, 128, 0, 0.2)",
            "rgba(255, 0, 0, 0.2)",
        ],
        "si": {
            "unit": "°C",
            "range": [10, 33.5],
        },
        "ip": {
            "unit": "°F",
            "range": [50, 92.5],  # ! needs update
        },
        "conversion_function": "temperature",
    },
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
    "UTCI": {
        "name": "UTCI",
        "order_categories": [
            "Out of<br>Range",
            "Extreme<br>Heat Stress",
            "Very Strong<br>Heat Stress",
            "Strong<br>Heat Stress",
            "Moderate<br>Heat Stress",
            "No Thermal<br>Stress",
            "Slight<br>Cold Stress",
            "Moderate<br>Cold Stress",
            "Strong<br>Cold Stress",
            "Very Strong<br>Cold Stress",
            "Extreme<br>Cold Stress",
        ],
        "colors": [
            "#5555ff",
            "#5f8dd3",
            "#80b3ff",
            "#87cdde",
            "#aaeeff",
            "#80d2a1",
            "#f4d7d7",
            "#e9afaf",
            "#ff8080",
            "#ff5555",
        ],
        "colors_categories": [
            "grey",  # Out of Range
            "#ff5555",  # Extreme Heat Stress
            "#ff8080",  # Very Strong Heat Stress
            "#e9afaf",  # Strong Heat Stress
            "#f4d7d7",  # Moderate Heat Stress
            "#80d2a1",  # No Thermal Stress
            "#aaeeff",  # Slight Cold Stress
            "#87cdde",  # Moderate Cold Stress
            "#80b3ff",  # Strong Cold Stress
            "#5f8dd3",  # Very Strong Cold Stress
            "#5555ff",  # Extreme Cold Stress
        ],
        "si": {
            "unit": "°C",
            "range": [-50, 50],
        },
        "ip": {
            "unit": "°F",
            "range": [-58, 122],
        },
        "conversion_function": "temperature",
    },
}
