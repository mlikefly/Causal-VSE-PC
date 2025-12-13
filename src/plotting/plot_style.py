from __future__ import annotations

import matplotlib


def apply_paper_style() -> None:
    """Apply a consistent paper-grade plotting style across the project.

    - Headless backend (Agg)
    - Seaborn-like clean theme with readable fonts
    - Thicker lines, subtle grids, consistent DPI handled by matplotlibrc
    """
    try:
        # Always use non-interactive backend for scripts
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

    try:
        import seaborn as sns
        # Clean white theme; grid only on y for readability
        sns.set_theme(
            context='paper',
            style='whitegrid',
            palette='deep',
            rc={
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.linestyle': '--',
                'grid.alpha': 0.25,
                'lines.linewidth': 2.0,
            },
        )
    except Exception:
        # Fallback: minimal rc tuning without seaborn
        from matplotlib import rcParams
        rcParams['axes.grid'] = True
        rcParams['grid.linestyle'] = '--'
        rcParams['grid.alpha'] = 0.25
        rcParams['lines.linewidth'] = 2.0






