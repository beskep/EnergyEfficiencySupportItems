from __future__ import annotations

import cyclopts
import matplotlib.pyplot as plt
import polars as pl

from eesi import utils
from eesi.config import BldgType, Config, Vars

app = utils.cli.App(
    config=cyclopts.config.Toml('config.toml', use_commands_as_keys=False)
)


def _upset_plot(
    data: pl.DataFrame | pl.LazyFrame,
    membership: str,
    value: str | None = None,
    *,
    max_subset: int = 20,
):
    import upsetplot  # noqa: PLC0415

    membership_data = (
        data.lazy()  # fmt
        .select(pl.col(membership))
        .collect()
        .to_series()
        .to_pandas()
    )
    upset_data = upsetplot.from_memberships(
        membership_data,
        data=None if value is None else data.lazy().select(value).collect().to_pandas(),
    )

    upset = upsetplot.UpSet(
        upset_data,
        subset_size='count',
        sort_by='cardinality',
        max_subset_rank=max_subset,
        show_counts='{:,}',
    )

    if value:
        upset.add_catplot(
            kind='violin', value=value, linewidth=0.75, density_norm='width'
        )

    return upset


@app.command
def upset(bldg: BldgType, *, conf: Config):
    # TODO 연도 구분
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
