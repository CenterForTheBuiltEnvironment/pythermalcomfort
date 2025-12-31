# Instructions for Claude Code

1. Review notes/development-context.md for a general overview of me and my colleagues were thinking earlier
2. Thoroughly review current implementation developed by my colleague: review current approach documented in pythermalcomfort/plots and pythermalcomfort/plots/matplotlib and use it as a starting point; I'd like to build on that and hear your feedback on how valuable this is moving forward; especially the models should be the primary source
3. Review earluer AI chatbot proposal in notes/claude-proposal.md
4. Develop a plan for the plotting feature that is a good compromise between flexibility and usability: the implemenation should be neat, concise and well-structured, but not overcomplicated and hard to use for intermediate users; I had the following thoughts on this: Ideally, we design the plotting functionality component-oriented: I was thinking that each plot could consist of three distinct elements: 1. Data representation: mapping data to a marker/visual, e.g. using dots, lines, ranges; 2. Context: This could sit on top of the data representation and provides the possibility to add to the plot over/underlays (comfort regions, psychrometric chart, category summary plot (e.g. counting number of samples in certain category)); this would depend on the model used; 3. Aesthetics: all settings that users might want to adjust: colors, font settings, axis styling, colormaps etc. (only as much as needed, doesn't to be too sophisticated) -> this No.3 should be the only component the user can adjust afterwards


IMPORTANT: This is about planning the implementation, so please focus on this. I think showing examples of the full stack would be great, but not for every single plot (similar to the proposal in notes/claude-proposal.md)


## Updated Instructions after Review of Phase 1

### General Feedback

Great initial work. I like the setup using dataclasses and also the way how the users can create plots using them and adjust the styling afterwards. I also like the Preset idea and would like to keep it for future iterations. I've thought a bit more about the setup. Please see my clarifying comments outlining my updated thoughts on how to implement this below.


### Readjustments for Phase 2 and setup moving forward

First of all, I want to clarify what these plots are for. This is not about writing custom functions for standard plots like bar charts, line plots, boxplots or similar. We will add some examples how to use pythermalcomfort and standard libraries like matplotlib for generating these "easy" and typical charts later. This plot feature focuses on simplifying the process of generating more complicated plots like psychrometric charts or others that e.g. rely on using a solver for ranges and use pythermalcomfort models. I think this is important to set at the start.

Regarding the overall, high-level structure, I feel like we should have three tiers/objects that setup a graph. This will introduce some changes to the current code structure:

1. Chart or Style object: Use it to set core figure styling like Figsize, Fonts, Title text, Axis styling, DPI etc., similar to what we do when we use plt.Figure. This is pretty much what you already have in style.py. Generally, the idea (compared to other object types) is that the users can modify these settings easily after the plot was generated.

2. Scene object or class (maybe to some degree a combination of the current Range and Region classes): This object sets up the scene/context for the plot and typically depends on pythermalcomfort model and a solver. Examples are a psychrometric chart or the chart for the UTCI ranges you generated earlier. The Scene generates a plot with "regions" or "ranges" displaying a category or comfort region for a given set up input parameters (for example x and y + additional fixed params if the underlying model requires more inputs). The Scene object can be an informative plot on its own or can be used as the scene/context/background on which a DataSeries object can be added (e.g. scatter points)

    Generally, users should be able to choose the two variables (x and y) and set the remaining fixed params/constraints, which depend on the model. Some models only require two variables (e.g. Heat Index, Adaptive), so they don't need fixed params. For others, like the PMV, the user should be able to use any two inputs as the variables on x and y, and remaining inputs are used as fixed params. Ideally, all variables have a default value, so that even when no fixed parameters are provided, the plot works by using the defined x and y and the default values for remaining inputs. To highlight the fixed values, there should be an annotation in the plot, that shows the fixed params (if any), e.g. in a subtitle -> with the option to turn it off.

    Different types of Scene objects could be (in order of priority for implementation):

    a. Generic range: Basically what you've done for the UTCI example in the test_notebook.ipynb. This is a generic Scene object that allows to visualize model output ranges (e.g. comfort zones) based on inputs for x and y + fixed params (if any required). I can see this applying to UTCI, the Heat Index and similar indices. You can build on what we already have.
    b. Adaptive: Similar to above, but specific to the Adaptive comfort model (please review example in pythermalcomfort/plots/earlier_examples and be aware that this also includes a DataSeries object which we don't need yet)
    c. Psychrometric: Similar to above, but a bit more specific since we need to generate the Psychrometric chart (please review example in pythermalcomfort/plots/earlier_examples and be aware that this also includes a DataSeries object which we don't need yet, but the use of the solver etc. could be similar)
    d. HOY: This is of low-priority for now. But I see this as a chart with days of the year on x and hours of the day on y that can be used as the scene to visualize a heatmap showing values for each hour of the year (another plot where I already have an earlier example (see heatmap), please review example in pythermalcomfort/plots/earlier_examples)
    e. Timeseries: Also of low-priority for now. This is slightly different than above, but can be helpful. Instead of having two inputs on x and y, this one would only need one input for y and could be the predicted index value (e.g. PMV). The idea here is that we set the Scene so the user can add a DataSeries object that contains timeseries data, e.g. hourly values for Heat Index over a week. The y-axis should use datetime format.

    All of them have unique requirements/need specific modifications, but still all have the same purpose: set the Scene for adding DataSeries objects and and at least for some of them, the option to stay as they are as a standalone, informative plot (Range, pmv, adaptive).

    For the styling (e.g. colormaps for ranges) of Scene object, I'd like to make use of your suggestions you use presets for each model that will be the default.

3. DataSeries: Add DataSeries that can be added on top of Scene. Styling should be adjustable through Style class. Examples are scatter points or heat mapped data. Ideally, when a DataSeries object is added, I'd also like to have the option to add a summary showing the distribution of the data points across categories in % of values. E.g. for PMV, a summary of how many data points are in "Neutral", "Slightly warm" etc. Again, you should be able to see what I did ealier in the examples folder (pythermalcomfort/plots/earlier_examples). I think I added some kind of summary there, too.

### Task List

1. Review earlier content and things added here. To me it is also important to honour the very first contributions by my colleague in plots/matplotlib and generic.py and heatmap_tdb_rh.py. I know he has put a lot of thought into this and I'd like to make use of his work where possible.
2. Review examples of earlier plots for pmv, adaptive and heatmap in pythermalcomfort/plots/earlier_examples. Please note: they were written in a different context, so there's no need to replicate exactly what I have done. However, the styling and looks of the plot worked out to be exactly what I wanted, so they can give you a hint where I want to go for these three plots.
3. Together with me, develop a plan how to implement this in an interative way. I want you to be honest with me and an indepedent thinker, that consider the user when suggesting solutions. It is very important, that we don't reinvent the wheel here and don't make too complicated. Remember: We want to help users generate the more complicated charts, e.g. for PMV or Adaptive chart, and don't want to make things even more complicated for them. So our solution should be clean, clear and not too sophisticated.
---

## Session Summary - December 31, 2024

### Phase 2 Implementation Complete

Implemented the three-tier plotting architecture:
1. **Style** (mutable) - Figure styling
2. **Scene** (frozen) - RangeScene, AdaptiveScene
3. **DataSeries** (frozen) - Overlay data with summary

### Key Files Created/Modified

**New files:**
- `plots/scenes/` directory with `base.py`, `range_scene.py`, `adaptive_scene.py`
- `plots/data_series.py` - DataSeries class
- `plots/summary.py` - SummaryRenderer (stacked horizontal bar)

**Modified:**
- `plots/style.py` - Added scatter, summary settings
- `plots/plot.py` - New factory methods, removed GridSpec
- `plots/ranges.py` - Added render() method (consolidated from regions.py)
- `plots/__init__.py` - Updated exports

**Removed:**
- `plots/regions.py` - Consolidated into ranges.py
- `plots/heatmap_tdb_rh.py` - Unused script

### Summary Bar Implementation

Changed from separate subplot to stacked horizontal bar as inset:
- Positioned at bottom-right using `inset_axes`
- Uses same colors as legend (no labels needed)
- Style settings: `summary_bar_width`, `summary_bar_height`, `summary_y_position`, `summary_min_pct_for_text`
- `render()` now always returns single axes (not list)

### API Examples

```python
# Basic usage
plot = Plot.range(utci, fixed_params={"v": 1.0, "tr": 25})
fig, ax = plot.render()

# With data and summary
plot = plot.add_data(temps, rh_vals)
plot.style.show_summary = True
fig, ax = plot.render()

# Adaptive comfort
plot = Plot.adaptive()
plot = plot.add_data(t_outdoor, t_operative)
fig, ax = plot.render()
```

### Tests
- All 241 tests pass
- All notebook cells verified working
