# Development Context Notes

## The package

pythermalcomfort [https://pythermalcomfort.readthedocs.io/en/latest/](https://pythermalcomfort.readthedocs.io/en/latest/) is a widely used Python package for calculating thermal comfort, heat stress, and cold stress indices. We aim to introduce new plotting functionalities that transform how these indices can be visualised and compared.

The new functions we plan to implement generate graphical representations of comfort regions across environmental parameters, such as air temperature and humidity, or on psychrometric charts. Users can visualise comfort zones using any model in pythermalcomfort (see pythermalcomfort/models/), for example, the PMV, SET, UTCI, PHS, and JOS3, enabling direct visual comparison of model outputs under identical conditions.

## Planned code implementation

### Code Implementation (Charting Functionality) requirements

* API Design: Review seaborn’s approach to coding style/structure and adopt a similar philosophy.
* Refactoring: Determine the best structure (e.g., one overarching ranges function vs. specific functions for t_rh, t_v, etc.).
* Atomicity: Ensure functions are atomic and robust. They should calculate/plot the data and return the axis object, leaving styling (labels, titles) to the user.
* Testing: Implement a robust suite of unit tests [OPTIONAL, can be done later]
* Features: Implement, at minimum, the charts found in the CBE Thermal Comfort Tool using matplotlib: Adaptive charts; Ranges (Psychrometric and X vs Y) with filled areas and borderlines; Thermal loss vs Environmental parameters [START with one or two plots first, take care of additional plots later]
* Examples: Expand on and create new plotting examples, in addition to the ones I have developed.


### Background Information for current implementation

#### Summary
A set of new plotting utilities for visualizing comfort and risk regions is available under pythermalcomfort.plots (branch: `plots/first_attempt`).
These provide reusable plotters such as `ranges_tdb_rh`, `ranges_tdb_v`, and a `psychrometric` helper.
The functions return Matplotlib Axes and artists so plots are composable and customizable.
It is a long message but I would really appreciate if you could read it and provide feedback, if possible answering the questions below.

#### Example

Simple example of a temperature vs relative humidity plot using the PMV/PPD model from ISO 7730 standard.
More examples and customization options are provided below.
```python
    from pythermalcomfort.models import pmv_ppd_iso

    ax, _ = ranges_tdb_rh(
        model_func=pmv_ppd_iso,
        fixed_params={"tr": 30, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(10, 36),
        rh_range=(0, 100),
    )
    plt.show()
```

#### Philosophy & Approach
1. Generic, reusable functions: core helpers like plot_threshold_region accept a model function and thresholds so they are model-agnostic.
2. Minimal formatting, maximum customization: functions return axes + artists; users can compose and tweak.
3. Consistent API across plotters: same param names for model_func, fixed_params, thresholds, plot_kwargs.
4. Colormap consistency: band rendering and small summary bars share the same colormap sampling. Use the Matplotlib 3.7+ API (matplotlib.colormaps["Name"].resampled(n)) to avoid deprecation.
5. Separation of concerns: computation and plotting are separated to ease testing and reuse.


---

## Research

* What about using [seaborn.objects](https://seaborn.pydata.org/api.html#objects-interface)? Seems to give me the flexibility that I want and let's user change the appearance of the plot easily
