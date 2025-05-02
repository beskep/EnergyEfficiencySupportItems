from __future__ import annotations

import dataclasses as dc
import functools
import itertools

import cyclopts
import matplotlib.pyplot as plt
import pingouin as pg
import polars as pl
import seaborn as sns
from cmap import Colormap

from eesi import utils
from eesi.config import BldgType, Config, ResidVars, Vars


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
def upset(bldg: BldgType, *, support_type: bool = False, conf: Config):
    lf = pl.scan_parquet(conf.source(bldg))
    upset = _upset_plot(
        lf.filter(
            pl.col(Vars.CONSTR).is_not_null(),
            pl.col(Vars.CONSTR).list.len() != 0,
        ),
        membership=Vars.CONSTR,
        value=Vars.COST,
    )
    fig = plt.figure()
    upset.plot(fig)
    fig.savefig(conf.dirs.analysis / f'0000.upset-{bldg}.png')
    plt.close(fig)

    if support_type and bldg == BldgType.RESIDENTIAL:
        upset = _upset_plot(
            lf.filter(pl.col(ResidVars.SUPPORT_TYPE).is_not_null()),
            membership=ResidVars.SUPPORT_TYPE,
            value=Vars.COST,
        )
        fig = plt.figure()
        upset.plot(fig)
        fig.savefig(conf.dirs.analysis / f'0000.upset-{bldg}-support.png')
        plt.close(fig)


@dc.dataclass
class _ResidentialTypes:
    conf: Config
    lf: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        self.lf = pl.scan_parquet(self.conf.source(BldgType.RESIDENTIAL)).select(
            Vars.COST, Vars.CONSTR, ResidVars.OWNERSHIP, ResidVars.RESIDENTIAL_TYPE
        )

    def heatmap_types_count(self):
        count = (
            self.lf.group_by(ResidVars.RESIDENTIAL_TYPE, ResidVars.OWNERSHIP)
            .len()
            .collect()
            .drop_nulls()
            .pivot(
                ResidVars.RESIDENTIAL_TYPE,
                index=ResidVars.OWNERSHIP,
                values='len',
                sort_columns=True,
            )
            .sort(ResidVars.OWNERSHIP)
            .fill_null(0)
            .to_pandas()
            .set_index(ResidVars.OWNERSHIP)
        )

        fig, ax = plt.subplots()
        sns.heatmap(
            count, annot=True, fmt=',', cmap=Colormap('cmasher:ocean').to_mpl(), ax=ax
        )
        ax.yaxis.set_tick_params(rotation=0)
        ax.set_ylabel('')
        fig.savefig(self.conf.dirs.analysis / '0001.주택유형.png')
        plt.close(fig)

    @functools.cached_property
    def _cost_palette(self):
        return tuple(Colormap('tol:bright')([0, 1, 4, 6]))

    def plot_cost(self, *, etc: bool = True, boiler: bool = True):
        data = (
            self.lf.filter(
                pl.lit(value=True)
                if boiler
                else pl.col(Vars.CONSTR).list.contains('보일러').not_()
            )
            .with_columns(pl.col(Vars.COST) / 10000)
            .drop(Vars.CONSTR)
            .collect()
        )

        if not etc:
            data = data.filter(
                pl.col(ResidVars.OWNERSHIP) != '기타',
                pl.col(ResidVars.RESIDENTIAL_TYPE) != '기타',
            )

        types = data[ResidVars.RESIDENTIAL_TYPE].unique().sort().to_list()
        ownership = ['자가', '전체무료임차', '비자가']
        if etc:
            ownership.append('기타')

        fig, ax = plt.subplots()
        sns.barplot(
            data,
            x=Vars.COST,
            y=ResidVars.RESIDENTIAL_TYPE,
            order=types,
            hue=ResidVars.OWNERSHIP,
            hue_order=ownership,
            ax=ax,
            palette=self._cost_palette,
            seed=42,
        )
        ax.legend(loc='upper left', framealpha=0.95).set_title('')
        ax.set_xlabel(f'{Vars.COST} (만원)')
        ax.set_ylabel('')

        fig.savefig(
            self.conf.dirs.analysis / '0001.유형별 지원액'
            f'{"" if etc else "(기타 제외)"}'
            f'{"" if boiler else "(보일러 제외)"}.png'
        )
        plt.close(fig)

    def _data(self, ownership: str, resid_type: str):
        lf = self.lf.with_columns(pl.col(Vars.COST) / 10000)  # 만원

        match ownership:
            case '전체':
                pass
            case '기타제외':
                lf = lf.filter(pl.col(ResidVars.OWNERSHIP) != '기타')
            case _:
                lf = lf.filter(pl.col(ResidVars.OWNERSHIP) == ownership)

        match resid_type:
            case '전체':
                pass
            case '기타제외':
                lf = lf.filter(pl.col(ResidVars.RESIDENTIAL_TYPE) != '기타')
            case _:
                lf = lf.filter(pl.col(ResidVars.RESIDENTIAL_TYPE) == resid_type)

        variables = [ResidVars.OWNERSHIP, ResidVars.RESIDENTIAL_TYPE]
        filters = [ownership, resid_type]
        between = [
            x
            for x, y in zip(variables, filters, strict=True)
            if y in {'전체', '기타제외'}
        ]

        return lf, between

    def _anova(
        self,
        ownership: str,
        resid_type: str,
    ) -> pl.DataFrame:
        lf, between = self._data(ownership, resid_type)

        anova = pg.anova(
            lf.collect().to_pandas(),
            dv=Vars.COST,
            between=between,
            ss_type=2,
            detailed=True,
            effsize='np2',
        )
        return pl.from_pandas(anova).select(
            pl.lit(ownership).alias(ResidVars.OWNERSHIP),
            pl.lit(resid_type).alias(ResidVars.RESIDENTIAL_TYPE),
            pl.all(),
        )

    def anova(self):
        def values(name):
            expr = pl.col(name)
            return (
                self.lf.select(expr.unique().sort())
                .filter(expr != '기타')
                .collect()
                .to_series()
            )

        ownership = values(ResidVars.OWNERSHIP)
        resid_type = values(ResidVars.RESIDENTIAL_TYPE)

        def cases():
            yield from itertools.product(['전체', '기타제외'], ['전체', '기타제외'])
            yield from itertools.product(ownership, ['전체', '기타제외'])
            yield from itertools.product(['전체', '기타제외'], resid_type)

        table = pl.concat(
            itertools.starmap(self._anova, cases()), how='vertical_relaxed'
        )
        table.write_excel(
            self.conf.dirs.analysis / '0001.가구 유형별 비용 ANOVA.xlsx',
            column_widths=150,
        )


@app.command
def residential_type(conf: Config):
    r = _ResidentialTypes(conf)
    r.anova()
    r.heatmap_types_count()

    for e, b in itertools.product([True, False], [True, False]):
        r.plot_cost(etc=e, boiler=b)


if __name__ == '__main__':
    utils.mpl.MplTheme(0.8).grid(color='0.75', lw=0.8).apply()
    app()
