from __future__ import annotations

import contextlib
import dataclasses as dc
import functools
import io
import itertools
from typing import TYPE_CHECKING, ClassVar

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import polars as pl
import polars.selectors as cs
import seaborn as sns
import shutup
import wand.image
from cmap import Colormap
from loguru import logger
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import PercentFormatter

from eesi import utils
from eesi.config import BldgType, Config, Vars
from eesi.utils._terminal import Progress

if TYPE_CHECKING:
    from pathlib import Path

    import marsilea as ma
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    # TODO to utils
    fig = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


def _trim_figure(fig: Figure, pad: int | None = 10):
    f = io.BytesIO()
    fig.savefig(f)
    image = wand.image.Image(blob=f.getvalue())
    image.trim()

    if pad:
        image.extent(
            width=image.width + 2 * pad,
            height=image.height + 2 * pad,
            gravity='center',
        )

    return image


app = utils.cli.App(
    config=cyclopts.config.Toml('config.toml', use_commands_as_keys=False)
)


@dc.dataclass
class _UpsetPlotter:
    conf: Config
    bldg: BldgType
    construction: str = Vars.CONSTR  # membership
    cost: str = Vars.COST
    max_subset: int = 20

    @functools.cached_property
    def _colormap(self):
        return Colormap('seaborn:crest')([0.2, 0.5, 0.8])

    def __call__(self, output: Path | None = None):
        output = output or self.conf.dirs.analysis
        name = f'{self.bldg}-{self.construction}-{self.cost}'

        with figure_context() as fig:
            self.by_upsetplot().plot(fig)
            _trim_figure(fig).save(filename=output / f'0000.upset-upsetplot-{name}.png')

        with figure_context() as fig:
            self.by_marsilea().render(fig)
            fig.savefig(output / f'0000.upset-marsilea-{name}.png')

    def _data(self):
        lf = pl.scan_parquet(self.conf.source(self.bldg)).filter(
            pl.col(self.construction).is_not_null(),
            pl.col(self.construction).list.len() != 0,
        )

        if self.bldg is BldgType.RESIDENTIAL:
            cost = pl.col(Vars.COST) - pl.col(Vars.COST_CONTRACTOR)
            lf = lf.with_columns(cost.alias(Vars.COST_MATERIAL))

        return lf

    def by_upsetplot(self):
        from eesi import _upset  # noqa: PLC0415

        df = (
            self._data()
            .select(self.construction, self.cost, Vars.YEAR)
            .collect()
            .to_pandas()
        )
        data = _upset.from_memberships(
            memberships=df[self.construction],
            data=df[[self.cost, Vars.YEAR]],
        )
        upset = _upset.UpSet(
            data,
            subset_size='count',
            sort_by='cardinality',
            max_subset_rank=self.max_subset or None,
            show_counts='{:,}',
            intersection_plot_elements=0,
        )
        upset.add_stacked_bars(Vars.YEAR, colors=self._colormap, title='연도별 건수')
        upset.add_catplot(
            kind='violin', value=self.cost, linewidth=0.75, density_norm='width'
        )
        return upset

    def by_marsilea(self) -> ma.upset.Upset:
        import marsilea as ma  # noqa: PLC0415

        df = (
            self._data()
            .select(self.construction, self.cost, Vars.YEAR)
            .with_columns(
                pl.len()
                .over(self.construction)
                .rank('dense', descending=True)
                .alias('rank')
            )
            .with_row_index()
            .collect()
            .to_pandas()
            .set_index('index')
        )

        data = ma.upset.UpsetData.from_memberships(
            items=df[self.construction],
            items_names=df.index,
            items_attrs=df[[self.cost, Vars.YEAR]],
        )
        if self.max_subset:
            cardinality = data.sets_table()['cardinality'].to_numpy()
            min_cardinality = (
                None
                if cardinality.size <= self.max_subset
                else np.sort(cardinality)[::-1][self.max_subset - 1]
            )

        upset = ma.upset.Upset(
            data,
            sort_sets='ascending',
            sort_subsets='-cardinality',
            min_cardinality=min_cardinality,
        )
        # TODO 연도별 개수
        upset.add_items_attr(
            side='top',
            attr_name=self.cost,
            plot='violin',
            pad=0.2,
            plot_kws={'density_norm': 'width'},
        )
        return upset


@app.command
def upset(*, conf: Config):
    utils.mpl.MplTheme(0.8, constrained=False).grid().apply()
    shutup.please()

    plotter = _UpsetPlotter(conf, bldg=BldgType.RESIDENTIAL)
    output = conf.dirs.analysis / '0000.upset'
    output.mkdir(exist_ok=True)

    def it():
        for bldg, const, cost in itertools.product(
            BldgType,
            [Vars.CONSTR, f'{Vars.CONSTR}(원본)'],
            [Vars.COST, Vars.COST_CONTRACTOR, Vars.COST_MATERIAL],
        ):
            if bldg is BldgType.SOCIAL_SERVICE and cost != Vars.COST:
                continue

            yield bldg, const, cost

    for bldg, const, cost in Progress.iter(list(it())):
        logger.info(f'{bldg=} | {const=} | {cost=}')

        plotter.bldg = bldg
        plotter.construction = const
        plotter.cost = cost
        plotter(output)


@dc.dataclass
class _ConstructionAnalysis:
    """유형별 시공 분포."""

    conf: Config
    VARIABLES: ClassVar[dict[BldgType, tuple[str, ...]]] = {
        BldgType.RESIDENTIAL: (Vars.Resid.OWNERSHIP, Vars.Resid.RESID_TYPE),
        BldgType.SOCIAL_SERVICE: (Vars.Social.STRATUM, Vars.Social.TARGET),
    }

    def analyse(
        self,
        bldg: BldgType,
        category: str,
        construction: str = Vars.CONSTR,
    ):
        data = (
            pl.scan_parquet(self.conf.source(bldg))
            .with_row_index()
            .select(
                'index',
                category,
                construction,
                pl.len().over(category).alias('total'),
            )
            .explode(construction)
            .drop_nulls(construction)
            .group_by(category, construction, 'total')
            .len('count')
            .with_columns(ratio=pl.col('count') / pl.col('total'))
            .sort(category, construction)
            .collect()
        )

        # 전체 평균 시공률 순서
        order = (
            data.group_by(construction)
            .agg(pl.sum('total', 'count'))
            .with_columns(ratio=pl.col('count') / pl.col('total'))
            .sort('ratio', descending=True)[construction]
        )

        col_wrap = utils.mpl.ColWrap(data[category].n_unique()).ncols

        grid = (
            sns.FacetGrid(data, col=category, col_wrap=col_wrap, height=3.2)
            .map_dataframe(sns.barplot, x='ratio', y=construction, order=order)
            .set_axis_labels('시공률', '')
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
        )

        grid.axes.ravel()[0].set_xlim(0, 1)

        percent_formatter = PercentFormatter(xmax=1)
        for ax in grid.axes_dict.values():
            ax.xaxis.set_major_formatter(percent_formatter)
            ax.bar_label(ax.containers[0], fmt='{:.1%}', padding=2)

        ConstrainedLayoutEngine().execute(grid.figure)

        return grid, data

    def __call__(self):
        d = self.conf.dirs.analysis / '0001.시공률'
        d.mkdir(exist_ok=True)

        dfs: list[pl.DataFrame] = []

        bldg: BldgType
        for bldg, construction in itertools.product(  # type: ignore[assignment]
            BldgType, [Vars.CONSTR, f'{Vars.CONSTR}(원본)']
        ):
            for var in self.VARIABLES[bldg]:
                grid, data = self.analyse(bldg, var, construction=construction)
                grid.savefig(d / f'0001.시공률-{bldg}-{var}-{construction}.png')
                plt.close(grid.figure)

                dfs.append(
                    data.select(
                        pl.lit(bldg).alias('building'),
                        pl.lit(var).alias('variable'),
                        pl.lit(construction).alias('construction'),
                        pl.all(),
                    ).rename(
                        {var: 'category', f'{Vars.CONSTR}(원본)': Vars.CONSTR},
                        strict=False,
                    )
                )

        (
            pl.concat(dfs)
            .sort('building', 'variable', 'construction', 'category')
            .write_excel(d.parent / '0001.시공률.xlsx', column_widths=120)
        )


@app.command
def construction(*, conf: Config):
    utils.mpl.MplTheme(0.8, constrained=False).grid().apply()
    _ConstructionAnalysis(conf)()


@dc.dataclass
class _ResidentialTypes:
    conf: Config
    lf: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        self.lf = pl.scan_parquet(self.conf.source(BldgType.RESIDENTIAL)).select(
            Vars.COST,
            Vars.CONSTR,
            cs.starts_with(Vars.Resid.OWNERSHIP),
            Vars.Resid.RESID_TYPE,
        )

    def plot_ownership(self):
        """원본 주거실태 개수."""
        v1 = Vars.Resid.OWNERSHIP
        v2 = f'{Vars.Resid.OWNERSHIP}(원본)'
        data = (
            self.lf.select(v1, v2).group_by(v1, v2).len('count').sort('count').collect()
        )

        fig, ax = plt.subplots()
        sns.barplot(data, x='count', y=v2, hue=v1, ax=ax)
        ax.set_xscale('log')
        ax.get_legend().set_title('주거실태 재분류')
        fig.savefig(self.conf.dirs.analysis / '0101.주거실태 재분류.png')

    def heatmap_types_count(self):
        count = (
            self.lf.group_by(Vars.Resid.RESID_TYPE, Vars.Resid.OWNERSHIP)
            .len()
            .collect()
            .drop_nulls()
            .pivot(
                Vars.Resid.RESID_TYPE,
                index=Vars.Resid.OWNERSHIP,
                values='len',
                sort_columns=True,
            )
            .sort(Vars.Resid.OWNERSHIP)
            .fill_null(0)
            .to_pandas()
            .set_index(Vars.Resid.OWNERSHIP)
        )

        fig, ax = plt.subplots()
        sns.heatmap(
            count, annot=True, fmt=',', cmap=Colormap('cmasher:ocean').to_mpl(), ax=ax
        )
        ax.yaxis.set_tick_params(rotation=0)
        ax.set_ylabel('')
        fig.savefig(self.conf.dirs.analysis / '0101.주택유형.png')
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
                pl.col(Vars.Resid.OWNERSHIP) != '기타',
                pl.col(Vars.Resid.RESID_TYPE) != '기타',
            )

        types = data[Vars.Resid.RESID_TYPE].unique().sort().to_list()
        ownership = ['자가', '전체무료임차', '비자가']
        if etc:
            ownership.append('기타')

        fig, ax = plt.subplots()
        sns.barplot(
            data,
            x=Vars.COST,
            y=Vars.Resid.RESID_TYPE,
            order=types,
            hue=Vars.Resid.OWNERSHIP,
            hue_order=ownership,
            ax=ax,
            palette=self._cost_palette,
            seed=42,
        )
        ax.legend(loc='upper left', framealpha=0.95).set_title('')
        ax.set_xlabel(f'{Vars.COST} (만원)')
        ax.set_ylabel('')

        fig.savefig(
            self.conf.dirs.analysis / '0101.유형별 지원액'
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
                lf = lf.filter(pl.col(Vars.Resid.OWNERSHIP) != '기타')
            case _:
                lf = lf.filter(pl.col(Vars.Resid.OWNERSHIP) == ownership)

        match resid_type:
            case '전체':
                pass
            case '기타제외':
                lf = lf.filter(pl.col(Vars.Resid.RESID_TYPE) != '기타')
            case _:
                lf = lf.filter(pl.col(Vars.Resid.RESID_TYPE) == resid_type)

        vars_ = [Vars.Resid.OWNERSHIP, Vars.Resid.RESID_TYPE]
        filters = [ownership, resid_type]
        between = [
            x for x, y in zip(vars_, filters, strict=True) if y in {'전체', '기타제외'}
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
            pl.lit(ownership).alias(Vars.Resid.OWNERSHIP),
            pl.lit(resid_type).alias(Vars.Resid.RESID_TYPE),
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

        ownership = values(Vars.Resid.OWNERSHIP)
        resid_type = values(Vars.Resid.RESID_TYPE)

        def cases():
            yield from itertools.product(['전체', '기타제외'], ['전체', '기타제외'])
            yield from itertools.product(ownership, ['전체', '기타제외'])
            yield from itertools.product(['전체', '기타제외'], resid_type)

        table = pl.concat(
            itertools.starmap(self._anova, cases()), how='vertical_relaxed'
        )
        table.write_excel(
            self.conf.dirs.analysis / '0101.가구 유형별 비용 ANOVA.xlsx',
            column_widths=150,
        )


@app.command
def plot_regional(*, conf: Config):
    for bldg in BldgType:
        data = (
            pl.scan_parquet(conf.source(bldg))
            .select(pl.col(Vars.COST) / 10000, Vars.RLG)
            .sort(Vars.RLG)
            .collect()
        )
        fig, ax = plt.subplots()
        sns.barplot(data, x=Vars.COST, y=Vars.RLG, ax=ax, seed=42)
        ax.set_xlabel(f'{Vars.COST} (만원)')
        ax.set_ylabel('')
        fig.savefig(conf.dirs.analysis / f'0002.지역별 지원액-{bldg}.png')
        plt.close(fig)


@app.command
def plot_residential(*, conf: Config):
    r = _ResidentialTypes(conf)

    r.anova()
    r.plot_ownership()
    r.heatmap_types_count()

    for e, b in itertools.product([True, False], [True, False]):
        r.plot_cost(etc=e, boiler=b)


@app.command
def plot_social(*, max_social_const: int = 8, conf: Config):
    # 사회복지시설 유형
    data = (
        pl.scan_parquet(conf.source(BldgType.SOCIAL_SERVICE))
        .select(pl.col(Vars.COST) / 10000, Vars.Social.STRATUM, Vars.Social.TARGET)
        .collect()
    )

    fig, axes = plt.subplots(1, 2, figsize=utils.mpl.MplFigSize(24, 9).inch())
    ax: Axes
    for var, elements, ax in zip(
        [Vars.Social.STRATUM, Vars.Social.TARGET],
        [Vars.Social.STRATUM_ELEMENTS, Vars.Social.TARGET_ELEMENTS],
        axes,
        strict=True,
    ):
        sns.violinplot(data, x=Vars.COST, y=var, order=elements, ax=ax)
        ax.set_xlabel(f'{Vars.COST} (만원)')
        ax.set_ylabel('')
        ax.set_title(f'{var}별 지원 금액')

    fig.savefig(conf.dirs.analysis / '0202.사회복지시설 유형별 지원액.png')
    plt.close(fig)

    # 사회복지시설 유형(대상)별 시공 유형
    const = f'{Vars.CONSTR}(원본)'
    data = (
        pl.scan_parquet(conf.source(BldgType.SOCIAL_SERVICE))
        .select(Vars.Social.TARGET, pl.col(const).list.sort().list.join(', '))
        .group_by(Vars.Social.TARGET, const)
        .len()
        .sort(Vars.Social.TARGET, 'len', descending=[False, True])
        .with_columns(
            pl.col('len')
            .rank('min', descending=True)
            .over(Vars.Social.TARGET)
            .alias('rank')
        )
        .filter(pl.col('rank') <= max_social_const)
        .collect()
    )
    grid = (
        sns.FacetGrid(
            data,
            col=Vars.Social.TARGET,
            col_order=Vars.Social.TARGET_ELEMENTS,
            col_wrap=2,
            sharey=False,
            height=3,
            aspect=16 / 9,
        )
        .map_dataframe(sns.barplot, x='len', y=const)
        .set_axis_labels('개수', '')
    )
    grid.savefig(conf.dirs.analysis / '0202.사회복지시설 유형별 시공.png')
    plt.close(grid.figure)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme(0.8).grid().apply()
    app()
