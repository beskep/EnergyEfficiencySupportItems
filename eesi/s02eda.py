from __future__ import annotations

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import rich
from cmap import Colormap

from eesi import utils
from eesi.config import BldgType, Config, Vars


def _upset_plot(
    data: pl.DataFrame | pl.LazyFrame,
    membership: str,
    value: str | None = None,
    *,
    max_subset: int = 20,
    by_year: bool = True,
):
    from eesi import _upset  # noqa: PLC0415

    membership_data = (
        data.lazy()  # fmt
        .select(pl.col(membership))
        .collect()
        .to_series()
        .to_pandas()
    )
    values = [Vars.YEAR] if value is None else [Vars.YEAR, value]
    upset_data = _upset.from_memberships(
        membership_data,
        data=data.lazy().select(values).collect().to_pandas(),
    )
    rich.print(upset_data)

    upset = _upset.UpSet(
        upset_data,
        subset_size='count',
        sort_by='cardinality',
        max_subset_rank=max_subset,
        show_counts='{:,}',
        intersection_plot_elements=0 if by_year else 6,
    )

    if by_year:
        upset.add_stacked_bars(
            Vars.YEAR,
            colors=Colormap('seaborn:crest')([0.2, 0.5, 0.8]),
            title='연도별 건수',
        )

    if value:
        upset.add_catplot(
            kind='violin', value=value, linewidth=0.75, density_norm='width'
        )

    return upset


app = utils.cli.App(
    config=cyclopts.config.Toml('config.toml', use_commands_as_keys=False)
)


@app.command
def upset(bldg: BldgType, *, conf: Config):
    lf = pl.scan_parquet(conf.dirs.data / f'0001.{bldg}.parquet')
    upset = _upset_plot(
        lf.filter(pl.col(Vars.CONSTR).is_not_null()),
        membership=Vars.CONSTR,
        value=Vars.COST,
    )
    fig = plt.figure()
    upset.plot(fig)
    fig.savefig(conf.dirs.analysis / f'0000.upset-{bldg}.png')
    plt.close(fig)

    if bldg == BldgType.RESIDENTIAL:
        upset = _upset_plot(
            lf.filter(pl.col(Vars.SUPPORT_TYPE).is_not_null()),
            membership=Vars.SUPPORT_TYPE,
            value=Vars.COST,
        )
        fig = plt.figure()
        upset.plot(fig)
        fig.savefig(conf.dirs.analysis / f'0000.upset-{bldg}-support.png')
        plt.close(fig)


# TODO 진단 가격 분포 체크


if __name__ == '__main__':
    utils.mpl.MplTheme(0.8).apply()
    app()
