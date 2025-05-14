from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import upsetplot
from matplotlib import cm
from upsetplot import from_contents, from_indicators, from_memberships, util

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Rectangle


__all__ = ['UpSet', 'from_contents', 'from_indicators', 'from_memberships']


class UpSet(upsetplot.UpSet):
    def _label_sizes_format(self, where):
        count_fmt = '{:.0f}' if self._show_counts is True else str(self._show_counts)

        if '{' not in count_fmt:
            count_fmt = util.to_new_pos_format(count_fmt)

        pct_fmt = '{:.1%}' if self._show_percentages is True else self._show_percentages

        def fmt(v: float):
            if v.is_integer():
                v = int(v)

            if count_fmt and pct_fmt:
                t = f'{count_fmt}{"\n" if where == "top" else ""}({pct_fmt})'
                return t.format(v, v / self.total)

            if count_fmt:
                return count_fmt.format(v)

            if pct_fmt:
                return pct_fmt.format(v / self.total)

            return ''

        return fmt

    def _label_sizes(
        self,
        ax: Axes,
        rects: Sequence[Rectangle],
        where: Literal['right', 'left', 'top'],
    ):
        if not (self._show_counts or self._show_percentages):
            return

        fmt = self._label_sizes_format(where)
        kwargs: dict = {
            'va': 'center',
            'ha': {'right': 'left', 'left': 'right', 'top': 'center'}[where],
            'size': 'x-small',
        }
        margin = 0.1 * float(
            np.abs(np.diff(ax.get_ylim() if where == 'top' else ax.get_xlim()))
        )

        for rect in rects:
            if where == 'top':
                value = rect.get_height() + rect.get_y()
                xy = (rect.get_x() + rect.get_width() * 0.5, value + margin)
            else:
                value = rect.get_width() + rect.get_x()
                xy = (value + margin, rect.get_y() + rect.get_height() * 0.5)

            ax.text(*xy, s=fmt(value), **kwargs)

    def _plot_stacked_bars(self, ax: Axes, by, sum_over, colors, title):
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
