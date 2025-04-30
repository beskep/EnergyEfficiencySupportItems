from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import upsetplot
from matplotlib import cm
from upsetplot import from_contents, from_indicators, from_memberships, util

if TYPE_CHECKING:
    from matplotlib.axes import Axes


__all__ = ['UpSet', 'from_contents', 'from_indicators', 'from_memberships']


class UpSet(upsetplot.UpSet):
    def _label_sizes_fmt(self, where):
        if self._show_counts is True:
            count_fmt = '{:.0f}'
        else:
            count_fmt = self._show_counts
            if '{' not in count_fmt:
                count_fmt = util.to_new_pos_format(count_fmt)

        pct_fmt = '{:.1%}' if self._show_percentages is True else self._show_percentages

        if count_fmt and pct_fmt:
            fmt = (
                f'{count_fmt}\n({pct_fmt})'
                if where == 'top'
                else f'{count_fmt} ({pct_fmt})'
            )

            def make_args(val):
                return val, val / self.total

        elif count_fmt:
            fmt = count_fmt

            def make_args(val):
                return (val,)

        else:
            fmt = pct_fmt

            def make_args(val):
                return (val / self.total,)

        return fmt, make_args

    def _label_sizes(self, ax: Axes, rects, where: Literal['right', 'left', 'top']):
        fmt, make_args = self._label_sizes_fmt(where)

        ha = {'right': 'left', 'left': 'right', 'top': 'center'}
        kwargs: dict = {'va': 'center', 'ha': ha[where]}
        margin: float = 0.1 * np.abs(
            np.diff(ax.get_ylim() if where == 'top' else ax.get_xlim())
        )

        def text(rect):
            value: float = (
                rect.get_height() + rect.get_y()
                if where == 'top'
                else rect.get_width() + rect.get_x()
            )
            if value.is_integer():
                value = int(value)

            if where == 'top':
                xy: tuple[float, float] = (
                    rect.get_x() + rect.get_width() * 0.5,
                    value + margin,
                )
            else:
                xy = (value + margin, rect.get_y() + rect.get_height() * 0.5)

            ax.text(*xy, s=fmt.format(*make_args(value)), **kwargs)

        for rect in rects:
            text(rect)

    def _plot_stacked_bars(self, ax, by, sum_over, colors, title):
        df: pd.DataFrame = self._df.set_index('_bin').set_index(
            by, append=True, drop=False
        )
        gb = df.groupby(level=list(range(df.index.nlevels)), sort=True)

        if sum_over is None and '_value' in df.columns:
            data = gb['_value'].sum()
        elif sum_over is None:
            data = gb.size()
        else:
            data = gb[sum_over].sum()

        data = data.unstack(by).fillna(0)
        if isinstance(colors, str):
            colors = cm.get_cmap(colors)
        elif isinstance(colors, Mapping):
            colors = data.columns.map(colors).values
            if pd.isna(colors).any():
                msg = (
                    'Some labels mapped by colors: '
                    f'{data.columns[pd.isna(colors)].tolist()!r}'
                )
                raise KeyError(msg)

        self._plot_bars(ax, data=data, colors=colors, title=title, use_labels=True)

        if self._horizontal:
            # Make legend order match visual stack order
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(list(reversed(handles)), list(reversed(labels)))
        else:
            ax.legend()

    PLOT_TYPES = upsetplot.UpSet.PLOT_TYPES | {
        'stacked_bars': _plot_stacked_bars,
    }
