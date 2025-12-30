# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pythermalcomfort is a Python package for calculating thermal comfort indices, heat/cold stress metrics, and thermophysiological responses. It implements standards like ASHRAE 55, ISO 7730, and EN 16798.

## Common Commands

```bash
# Run tests
pytest -q                              # Quick test run
pytest -k test_name_fragment           # Run specific tests
pytest --cov --cov-report=term-missing # With coverage
tox -e py312                           # Run single Python version

# Linting and formatting
ruff check --fix
ruff format
docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

# Full CI locally
tox
```

## Architecture

### Package Structure
- `pythermalcomfort/models/` - Each thermal comfort model in its own file (pmv_ppd_iso.py, utci.py, etc.)
- `pythermalcomfort/classes_input.py` - Input dataclasses with validation (BaseInputs pattern)
- `pythermalcomfort/classes_return.py` - Return type dataclasses
- `pythermalcomfort/utilities.py` - Shared utility functions
- `pythermalcomfort/plots/` - Plotting functionality (optional dependency, see below)
- `pythermalcomfort/jos3_functions/` - JOS-3 multinode model internals

### Key Patterns
- **Input validation**: Use dataclasses with `__post_init__` for validation. Inherit from `BaseInputs` when appropriate. Use `validate_type()` helper.
- **Vectorization**: Use numpy for all numerical operations (`np.log`, `np.asarray`). Support both scalar and array inputs.
- **Type hints**: Required on all functions.
- **Docstrings**: NumPy-style with Args, Returns, Raises, Examples sections. Include units and applicability limits.
- **Exceptions**: `TypeError` for wrong types, `ValueError` for invalid values.

### Adding a New Model
1. Create `pythermalcomfort/models/<name>.py`
2. Add input dataclass to `classes_input.py` with validation
3. Add return dataclass to `classes_return.py` if needed
4. Export from `models/__init__.py`
5. Add tests in `tests/tests_<name>.py`
6. Add `.. autofunction::` entry in docs

## Dependencies

Core: numpy, scipy, numba
Optional: matplotlib, pandas, seaborn (for plots extra)

Install with plots: `pip install pythermalcomfort[plots]`

## Plotting Feature (plots/)

Component-oriented architecture with three layers:
- **Ranges** (frozen): Data for threshold curves computed from models
- **Regions** (frozen): Context layer that renders Ranges as color bands
- **Style** (mutable): The ONLY component modifiable after plot creation

Key files:
- `plots/plot.py` - `Plot` facade class with `ranges()` factory method
- `plots/style.py` - `Style` dataclass for appearance settings
- `plots/ranges.py` - `Ranges` dataclass with `from_model()` factory
- `plots/regions.py` - `Regions` dataclass with `render()` method
- `plots/presets.py` - Model presets (PMV_PRESET, UTCI_PRESET, etc.)

Usage:
```python
from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots import Plot

plot = Plot.ranges(pmv_ppd_iso, fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0})
plot.style.title = "My Plot"  # Only style is mutable
fig, ax = plot.render()
```

Future work (Phase 2): Scatter data support, psychrometric charts, add_scatter() method.

## Branch Naming
- Features: `Feature/your-feature-name`
- Bugfixes: `Fix/your-bug-name`
- Docs: `Documentation/doc-name`
