Plots
=====

This section provides an overview of the plotting capabilities available in pythermalcomfort.
The library offers a variety of functions to visualize the results of the model calculations.
We also provide examples of how to create visually compelling and informative plots using popular Python libraries such as Matplotlib and Seaborn.

To use the plotting functions we included in pythermalcomfort, please install the optional dependencies using:

.. code-block:: bash

    pip install pythermalcomfort[plots]

Typical Use Cases
-----------------

- Comparing the results of different thermal comfort models.
- Visualizing the impact of various parameters (e.g., temperature, humidity, clothing insulation) on model output.
- Creating custom plots to communicate findings effectively.
- Analyse the comfort levels in different environments.
- Process, analyze, and visualise:
    - climate data in different formats (e.g., CSV, Excel, JSON, EPW)
    - results from building simulation software.
    - results from sensor data.

pythermalcomfort - Plotting Functions
-------------------------------------

The `plot` module in pythermalcomfort includes several built-in functions to create plots.
These function can be used to compare the results of different models or to visualize how various parameters affect their outcomes.
We the `plot` module has a sub-module for different backends, currently supporting Matplotlib only, but we plan to add support for other libraries in the future (e.g., Plotly, Bokeh, Seaborn).
All plotting functions return a figure object, which can be further customized if needed.
This allows users to create more complex visualizations by adding additional elements or modifying existing ones.

.. toctree::

    plots/matplotlib/plots.ipynb

Plotting Examples
-----------------

In addition to the built-in plotting functions, the library provides several examples demonstrating how to create custom plots using Matplotlib and Seaborn.
These examples cover a range of scenarios, from simple plots to more complex visualizations like heatmaps.
Users can adapt these examples to suit their specific needs and create visually appealing representations.

.. toctree::

    plots/matplotlib/utci.ipynb
