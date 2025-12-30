```bash
# index_plot.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class PlotTheme:
    """Theme configuration for plots."""
    figure_size: Tuple[float, float] = (10, 8)
    font_size: Dict[str, int] = field(default_factory=lambda: {
        'title': 16,
        'axis_label': 14,
        'tick': 12,
        'legend': 12
    })
    font_family: str = 'sans-serif'
    zone_alpha: float = 0.3
    point_size: float = 60
    point_color: str = 'black'
    point_alpha: float = 0.8
    grid: bool = True
    grid_alpha: float = 0.3
    zone_colors: Optional[Dict[str, str]] = None


@dataclass
class IndexConfiguration:
    """
    Configuration for a specific index plot.
    This defines the "rules" for how to plot a particular index.
    """
    name: str
    calculation_func: Callable
    zones: Dict[str, Tuple[float, float]]
    zone_colors: Dict[str, str]
    zone_labels: Dict[str, str]
    x_param: str
    y_param: str
    default_x_label: str
    default_y_label: str
    default_title: str
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    background_style: str = 'contour'  # 'contour', 'bands_horizontal', 'bands_vertical'
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure zones, colors, and labels have matching keys
        if set(self.zones.keys()) != set(self.zone_colors.keys()):
            raise ValueError("Zone names must match between zones and zone_colors")
        if set(self.zones.keys()) != set(self.zone_labels.keys()):
            raise ValueError("Zone names must match between zones and zone_labels")


class IndexPlot:
    """
    Generic index plotter that works with any IndexConfiguration.
    """
    
    def __init__(self, 
                 config: IndexConfiguration,
                 theme: Optional[PlotTheme] = None):
        """
        Initialize IndexPlot.
        
        Parameters
        ----------
        config : IndexConfiguration
            Configuration defining the index and how to plot it
        theme : PlotTheme, optional
            Theme configuration. If None, uses default theme.
        """
        self.config = config
        self.theme = theme if theme is not None else PlotTheme()
        
        # Override theme colors with config colors if theme doesn't specify
        if self.theme.zone_colors is None:
            self.theme.zone_colors = self.config.zone_colors.copy()
        
        self.fig = None
        self.ax = None
    
    def _create_contour_background(self, x_range: Tuple[float, float],
                                   y_range: Tuple[float, float],
                                   resolution: int = 100):
        """Create contour-based background showing index zones."""
        # Create mesh grid
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        
        # Calculate index for grid
        # Pass grid values as keyword arguments matching the config parameter names
        kwargs = {self.config.x_param: X_grid, self.config.y_param: Y_grid}
        Z_grid = self.config.calculation_func(**kwargs)
        
        # Create levels based on zones
        levels = sorted([vmin for _, (vmin, vmax) in self.config.zones.items()] + 
                       [vmax for _, (vmin, vmax) in self.config.zones.items()])
        levels = sorted(list(set(levels)))
        
        # Create color list in order
        zone_list = sorted(self.config.zones.items(), key=lambda x: x[1][0])
        color_list = [self.theme.zone_colors[name] for name, _ in zone_list]
        
        # Plot contour fill
        contour = self.ax.contourf(X_grid, Y_grid, Z_grid,
                                   levels=levels,
                                   colors=color_list,
                                   alpha=self.theme.zone_alpha,
                                   zorder=0)
        
        return contour
    
    def _create_band_background(self, orientation: str = 'horizontal'):
        """Create simple band background."""
        for zone_name, (vmin, vmax) in self.config.zones.items():
            color = self.theme.zone_colors.get(zone_name, '#cccccc')
            
            if orientation == 'horizontal':
                self.ax.axhspan(vmin, vmax, facecolor=color, 
                               alpha=self.theme.zone_alpha, zorder=0)
            else:  # vertical
                self.ax.axvspan(vmin, vmax, facecolor=color,
                               alpha=self.theme.zone_alpha, zorder=0)
    
    def plot(self,
             x_data: Union[List, np.ndarray],
             y_data: Union[List, np.ndarray],
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             show_legend: bool = True,
             x_range: Optional[Tuple[float, float]] = None,
             y_range: Optional[Tuple[float, float]] = None,
             background_style: Optional[str] = None):
        """
        Create index plot with background zones and data points.
        
        Parameters
        ----------
        x_data : array-like
            X-axis data values
        y_data : array-like
            Y-axis data values
        xlabel : str, optional
            X-axis label. If None, uses config default
        ylabel : str, optional
            Y-axis label. If None, uses config default
        title : str, optional
            Plot title. If None, uses config default
        show_legend : bool
            Whether to show legend for zones
        x_range : Tuple[float, float], optional
            X-axis range. If None, inferred from config or data
        y_range : Tuple[float, float], optional
            Y-axis range. If None, inferred from config or data
        background_style : str, optional
            Override config background style
        
        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        # Convert to numpy arrays
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        # Use defaults from config if not provided
        xlabel = xlabel or self.config.default_x_label
        ylabel = ylabel or self.config.default_y_label
        title = title or self.config.default_title
        background_style = background_style or self.config.background_style
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.theme.figure_size)
        
        # Determine ranges
        if x_range is None:
            x_range = self.config.x_range or (x_data.min() - 5, x_data.max() + 5)
        if y_range is None:
            y_range = self.config.y_range or (y_data.min() - 5, y_data.max() + 5)
        
        # Create background zones
        if background_style == 'contour':
            self._create_contour_background(x_range, y_range)
        elif background_style == 'bands_horizontal':
            self._create_band_background(orientation='horizontal')
        elif background_style == 'bands_vertical':
            self._create_band_background(orientation='vertical')
        
        # Plot data points
        self.ax.scatter(x_data, y_data,
                       s=self.theme.point_size,
                       c=self.theme.point_color,
                       alpha=self.theme.point_alpha,
                       zorder=5,
                       edgecolors='white',
                       linewidth=0.5)
        
        # Styling
        self.ax.set_xlabel(xlabel, fontsize=self.theme.font_size['axis_label'],
                          family=self.theme.font_family)
        self.ax.set_ylabel(ylabel, fontsize=self.theme.font_size['axis_label'],
                          family=self.theme.font_family)
        self.ax.set_title(title, fontsize=self.theme.font_size['title'],
                         family=self.theme.font_family, pad=20)
        
        self.ax.tick_params(labelsize=self.theme.font_size['tick'])
        
        # Set ranges
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        
        # Grid
        if self.theme.grid:
            self.ax.grid(True, alpha=self.theme.grid_alpha, zorder=1)
        
        # Legend
        if show_legend:
            # Sort zones by their minimum value for logical ordering
            sorted_zones = sorted(self.config.zones.items(), 
                                key=lambda x: x[1][0])
            legend_patches = [
                mpatches.Patch(color=self.theme.zone_colors[zone_name],
                             label=self.config.zone_labels[zone_name],
                             alpha=self.theme.zone_alpha)
                for zone_name, _ in sorted_zones
            ]
            self.ax.legend(handles=legend_patches,
                          loc='upper left',
                          fontsize=self.theme.font_size['legend'],
                          framealpha=0.9)
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def save(self, filename: str, dpi: int = 300, **kwargs):
        """Save the figure to file."""
        if self.fig is None:
            raise ValueError("No plot has been created yet. Call plot() first.")
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
    
    def show(self):
        """Display the plot."""
        if self.fig is None:
            raise ValueError("No plot has been created yet. Call plot() first.")
        plt.show()


# Convenience function
def plot_index(config: IndexConfiguration,
               x_data: Union[List, np.ndarray],
               y_data: Union[List, np.ndarray],
               theme: Union[str, PlotTheme] = 'default',
               **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick plot function for any index visualization.
    
    Parameters
    ----------
    config : IndexConfiguration
        Configuration for the index to plot
    x_data : array-like
        X-axis data values
    y_data : array-like
        Y-axis data values
    theme : str or PlotTheme
        Theme to use. Can be 'default', 'colorblind', or a PlotTheme instance
    **kwargs
        Additional arguments passed to IndexPlot.plot()
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Handle preset themes
    if isinstance(theme, str):
        if theme == 'colorblind':
            theme_obj = PlotTheme(
                zone_colors=None  # Will use config colors, just using CB-friendly defaults
            )
        else:  # default
            theme_obj = PlotTheme()
    else:
        theme_obj = theme
    
    plotter = IndexPlot(config=config, theme=theme_obj)
    return plotter.plot(x_data, y_data, **kwargs)


# ============================================================================
# INDEX CONFIGURATIONS
# ============================================================================

def calculate_heat_index(temperature_f: np.ndarray,
                        relative_humidity: np.ndarray) -> np.ndarray:
    """Calculate heat index using the NWS formula."""
    T = temperature_f
    RH = relative_humidity
    
    simple_hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    
    c1, c2, c3 = -42.379, 2.04901523, 10.14333127
    c4, c5, c6 = -0.22475541, -0.00683783, -0.05481717
    c7, c8, c9 = 0.00122874, 0.00085282, -0.00000199
    
    full_hi = (c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + c6*RH**2 +
               c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2)
    
    heat_index = np.where(T < 80, simple_hi, full_hi)
    return heat_index


HEAT_INDEX_CONFIG = IndexConfiguration(
    name='heat_index',
    calculation_func=calculate_heat_index,
    zones={
        'safe': (0, 80),
        'caution': (80, 91),
        'extreme_caution': (91, 103),
        'danger': (103, 125),
        'extreme_danger': (125, 200)
    },
    zone_colors={
        'safe': '#4caf50',
        'caution': '#ffeb3b',
        'extreme_caution': '#ff9800',
        'danger': '#f44336',
        'extreme_danger': '#880e4f'
    },
    zone_labels={
        'safe': 'Safe',
        'caution': 'Caution',
        'extreme_caution': 'Extreme Caution',
        'danger': 'Danger',
        'extreme_danger': 'Extreme Danger'
    },
    x_param='temperature_f',
    y_param='relative_humidity',
    default_x_label='Temperature (°F)',
    default_y_label='Relative Humidity (%)',
    default_title='Heat Index',
    x_range=(65, 115),
    y_range=(0, 100),
    background_style='contour'
)


# Example: Wind Chill Index
def calculate_wind_chill(temperature_f: np.ndarray,
                        wind_speed_mph: np.ndarray) -> np.ndarray:
    """Calculate wind chill using the NWS formula."""
    T = temperature_f
    V = wind_speed_mph
    
    # Formula only valid for T <= 50°F and V >= 3 mph
    wind_chill = 35.74 + 0.6215*T - 35.75*(V**0.16) + 0.4275*T*(V**0.16)
    
    # Where formula doesn't apply, return temperature
    wind_chill = np.where((T > 50) | (V < 3), T, wind_chill)
    
    return wind_chill


WIND_CHILL_CONFIG = IndexConfiguration(
    name='wind_chill',
    calculation_func=calculate_wind_chill,
    zones={
        'low': (-100, -19),
        'moderate': (-19, 0),
        'elevated': (0, 20),
        'minimal': (20, 100)
    },
    zone_colors={
        'low': '#1a237e',
        'moderate': '#42a5f5',
        'elevated': '#90caf9',
        'minimal': '#e3f2fd'
    },
    zone_labels={
        'low': 'Extreme Cold',
        'moderate': 'Very Cold',
        'elevated': 'Cold',
        'minimal': 'Cool'
    },
    x_param='temperature_f',
    y_param='wind_speed_mph',
    default_x_label='Temperature (°F)',
    default_y_label='Wind Speed (mph)',
    default_title='Wind Chill Index',
    x_range=(-30, 50),
    y_range=(0, 60),
    background_style='contour'
)


# Example: Air Quality Index (simplified)
def calculate_aqi(pm25: np.ndarray, ozone: np.ndarray) -> np.ndarray:
    """Simplified AQI calculation (normally more complex)."""
    # This is a simplified example - real AQI is more complex
    # Using PM2.5 as primary component
    aqi = pm25 * 3  # Simplified conversion
    return aqi


AQI_CONFIG = IndexConfiguration(
    name='air_quality_index',
    calculation_func=calculate_aqi,
    zones={
        'good': (0, 50),
        'moderate': (50, 100),
        'unhealthy_sensitive': (100, 150),
        'unhealthy': (150, 200),
        'very_unhealthy': (200, 300),
        'hazardous': (300, 500)
    },
    zone_colors={
        'good': '#00e400',
        'moderate': '#ffff00',
        'unhealthy_sensitive': '#ff7e00',
        'unhealthy': '#ff0000',
        'very_unhealthy': '#8f3f97',
        'hazardous': '#7e0023'
    },
    zone_labels={
        'good': 'Good',
        'moderate': 'Moderate',
        'unhealthy_sensitive': 'Unhealthy for Sensitive Groups',
        'unhealthy': 'Unhealthy',
        'very_unhealthy': 'Very Unhealthy',
        'hazardous': 'Hazardous'
    },
    x_param='pm25',
    y_param='ozone',
    default_x_label='PM2.5 (μg/m³)',
    default_y_label='Ozone (ppb)',
    default_title='Air Quality Index',
    x_range=(0, 150),
    y_range=(0, 150),
    background_style='contour'
)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Example 1: Heat Index
    print("Creating heat index plot...")
    n_points = 50
    temps = np.random.uniform(70, 110, n_points)
    humidity = np.random.uniform(20, 90, n_points)
    
    fig1, ax1 = plot_index(HEAT_INDEX_CONFIG, temps, humidity)
    plt.savefig('generic_heat_index.png', dpi=300)
    plt.close()
    
    # Example 2: Wind Chill with custom theme
    print("Creating wind chill plot...")
    temps_cold = np.random.uniform(-20, 45, n_points)
    wind_speed = np.random.uniform(5, 50, n_points)
    
    custom_theme = PlotTheme(
        figure_size=(12, 9),
        point_color='darkred',
        point_size=80,
        font_size={'title': 18, 'axis_label': 15, 'tick': 13, 'legend': 13}
    )
    
    fig2, ax2 = plot_index(WIND_CHILL_CONFIG, temps_cold, wind_speed,
                           theme=custom_theme)
    plt.savefig('generic_wind_chill.png', dpi=300)
    plt.close()
    
    # Example 3: Air Quality Index
    print("Creating air quality index plot...")
    pm25_vals = np.random.uniform(5, 120, n_points)
    ozone_vals = np.random.uniform(10, 130, n_points)
    
    fig3, ax3 = plot_index(AQI_CONFIG, pm25_vals, ozone_vals)
    plt.savefig('generic_aqi.png', dpi=300)
    plt.close()
    
    # Example 4: Using the class directly for more control
    print("Creating custom styled plot...")
    plotter = IndexPlot(
        config=HEAT_INDEX_CONFIG,
        theme=PlotTheme(
            figure_size=(14, 10),
            zone_alpha=0.5,
            point_color='navy',
            grid=True,
            grid_alpha=0.4
        )
    )
    
    fig4, ax4 = plotter.plot(
        temps, humidity,
        title="Custom Heat Index Analysis",
        background_style='contour'
    )
    plotter.save('generic_custom_style.png')
    plt.close()
    
    print("All plots created successfully!")
    ```