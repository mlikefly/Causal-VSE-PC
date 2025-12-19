# -*- coding: utf-8 -*-
"""
绘图样式配置模块。

提供论文级绘图样式的统一配置。
"""

from __future__ import annotations

import matplotlib


def apply_paper_style() -> None:
    """在整个项目中应用一致的论文级绘图样式。

    - 无头后端 (Agg)
    - 类似 Seaborn 的简洁主题，字体可读
    - 较粗的线条、微妙的网格，DPI 由 matplotlibrc 处理
    """
    try:
        # 脚本始终使用非交互式后端
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

    try:
        import seaborn as sns
        # 简洁白色主题；仅在 y 轴显示网格以提高可读性
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
        # 回退：不使用 seaborn 的最小 rc 调整
        from matplotlib import rcParams
        rcParams['axes.grid'] = True
        rcParams['grid.linestyle'] = '--'
        rcParams['grid.alpha'] = 0.25
        rcParams['lines.linewidth'] = 2.0






