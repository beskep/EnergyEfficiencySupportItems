from __future__ import annotations

import dataclasses as dc
import functools
import io
import itertools
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import cyclopts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
import shutup
import squarify
import wand.image
from cmap import Colormap
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import PercentFormatter

from eesi import utils
from eesi.config import BldgType, Config, Vars
from eesi.utils.terminal import Progress

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import marsilea as ma
    from matplotlib.axes import Axes


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

        with utils.mpl.figure_context() as fig:
            self.by_upsetplot().plot(fig)
            _trim_figure(fig).save(filename=output / f'0000.upset-upsetplot-{name}.png')

        with utils.mpl.figure_context() as fig:
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

    class _VarOrder(NamedTuple):
        var: str
        order: Sequence[str] | None

    @staticmethod
    def _grid(
        data: pl.DataFrame,
        y: _VarOrder,
        col: _VarOrder,
        *,
        threshold: float = 0.25,
    ):
        col_wrap = utils.mpl.ColWrap(data[col.var].n_unique()).ncols

        grid = (
            sns.FacetGrid(data, col=col.var, col_wrap=col_wrap, col_order=col.order)
            .map_dataframe(
                sns.barplot,
                x='supported',
                y=y.var,
                order=y.order,
                err_kws={'alpha': 0.75},
                seed=42,
            )
            .set_axis_labels('시공률', '')
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500, size='large')
        )

        grid.axes.ravel()[0].set_xlim(0, 1)

        percent_formatter = PercentFormatter(xmax=1)
        label_style = {'fmt': '{:.1%}', 'weight': 500, 'size': 'small'}
        center_color = utils.mpl.text_color(sns.color_palette(n_colors=1)[0])

        for ax in grid.axes_dict.values():
            ax.xaxis.set_major_formatter(percent_formatter)
            container: Any = ax.containers[0]
            centers = ax.bar_label(
                container, label_type='center', color=center_color, **label_style
            )
            edges = ax.bar_label(
                container, label_type='edge', padding=10, **label_style
            )

            for center, edge in zip(centers, edges, strict=True):
                value = float(center.get_text().removesuffix('%')) / 100

                if value < threshold:
                    center.remove()
                else:
                    edge.remove()

        grid.figure.set_size_inches(mpl.rcParams['figure.figsize'])
        ConstrainedLayoutEngine().execute(grid.figure)

        return grid

    def analyse(
        self,
        bldg: BldgType,
        category: str,
        construction: str = Vars.CONSTR,
    ):
        data = (
            pl.scan_parquet(self.conf.source(bldg))
            .with_row_index()
            # NOTE 시공 내역이 비었으나 지원 금액이 존재하는 record가 있음
            .drop_nulls(f'{Vars.CONSTR}(원본)')
            .select('index', category, construction)
            .explode(construction)
            .collect()
        )

        # index, category 조합
        cross = (
            data.select('index')
            .unique()
            .join(data.select(pl.col(construction).drop_nulls()).unique(), how='cross')
        )

        # 시공 지원 여부(supported)를 0, 1로 표시
        binary = (
            cross.join(data.select('index', category).unique(), on='index', how='left')
            .join(
                data.select('index', construction)
                .unique()
                .with_columns(pl.lit(1).alias('supported')),
                on=['index', construction],
                how='left',
            )
            .with_columns(pl.col('supported').fill_null(0))
        )

        ratio = (
            binary.group_by(category, construction)
            .agg(pl.mean('supported').alias('지원률'))
            .sort(pl.all())
        )

        def order(var: str):
            return (
                binary.group_by(var)
                .agg(pl.mean('supported'))
                .sort('supported', descending=True)[var]
                .to_list()
            )

        cat_order = self._VarOrder(category, order=order(category))
        const_order = self._VarOrder(construction, order=order(construction))
        grids = {
            'category': self._grid(binary, y=const_order, col=cat_order),
            'construction': self._grid(binary, y=cat_order, col=const_order),
        }

        return ratio, grids

    def _keys(self):
        bldg: BldgType
        for bldg, const in itertools.product(  # type: ignore[assignment]
            BldgType, [Vars.CONSTR, f'{Vars.CONSTR}(원본)']
        ):
            for var in self.VARIABLES[bldg]:
                yield bldg, const, var

    def __call__(self):
        d = self.conf.dirs.analysis / '0001.시공률'
        d.mkdir(exist_ok=True)

        def it():
            for bldg, const, var in Progress.iter(list(self._keys())):
                logger.info(f'{bldg=} | {const=} | {var=}')

                data, grids = self.analyse(bldg, var, const)

                for col, grid in grids.items():
                    grid.savefig(d / f'0001.시공률-{bldg}-{var}-{const}-{col}.png')
                    plt.close(grid.figure)

                yield (
                    data.select(
                        pl.lit(bldg).alias('building'),
                        pl.lit(var).alias('variable'),
                        pl.lit(const).alias('construction'),
                        pl.all(),
                    )
                    .rename(
                        {var: 'category', f'{Vars.CONSTR}(원본)': Vars.CONSTR},
                        strict=False,
                    )
                    .with_columns()
                )

        (
            pl.concat(it())
            .sort('building', 'variable', 'construction', 'category')
            .write_excel(d.parent / '0001.시공률.xlsx', column_widths=120)
        )


@app.command
def construction(*, width: float = 24, aspect=9 / 16, conf: Config):
    """유형별 시공률."""
    (
        utils.mpl.MplTheme(
            0.8,
            constrained=False,
            fig_size=(width, None, aspect),
            rc={'lines.solid_capstyle': 'projecting'},
        )
        .grid()
        .apply()
    )
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

    def plot_types_count(self):
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

    @staticmethod
    def _plot_const_ratio(
        data: pl.DataFrame,
        types: Sequence[str],
        output: Path,
        threshold: float = 0.05,
    ):
        data_etc = (
            data.with_columns(
                pl.when(pl.col('ratio') < threshold)
                .then(pl.lit('기타'))
                .otherwise(pl.col(Vars.CONSTR))
                .alias(Vars.CONSTR)
            )
            .group_by([*types, Vars.CONSTR, 'total'])
            .agg(pl.sum('count'))
            .with_columns(ratio=pl.col('count') / pl.col('total'))
            .with_columns(
                label=pl.format(
                    '{}\n({}%)', Vars.CONSTR, pl.col('ratio').mul(100).round(1)
                ),
                key=pl.when(pl.col(Vars.CONSTR) == '기타')
                .then(pl.lit(-1))
                .otherwise(pl.col('ratio')),
            )
            .sort([*types, 'key'], descending=[False, False, True])
        )

        pie_colors = tuple(Colormap('tol:light-alt').color_stops.color_array)
        tree_cmap = Colormap('cmasher:ocean')
        rng = np.random.default_rng(seed=42)

        for t, df in Progress.iter(
            data_etc.group_by(types, maintain_order=True),
            total=data_etc.select(types).n_unique(),
        ):
            name = '-'.join(map(str, t))

            # pie
            fig = Figure()
            ax = fig.add_subplot()
            ax.pie(
                df['ratio'].to_numpy(),
                labels=df['label'].to_list(),
                startangle=90,
                colors=pie_colors,
                counterclock=False,
                wedgeprops={'alpha': 0.75},
            )
            _trim_figure(fig).save(filename=output / f'pie-{name}.png')

            # treemap
            fig = Figure()
            ax = fig.add_subplot()
            colors = tree_cmap(rng.uniform(0.5, 0.8, df.height))
            rng.uniform()
            squarify.plot(
                sizes=df['count'],
                label=df['label'],
                color=colors,
                ax=ax,
                bar_kwargs={'alpha': 0.9},
                text_kwargs={'weight': 500},
            )
            ax.set_axis_off()
            fig.savefig(output / f'treemap-{name}.png')

    def plot_const_ratio(self):
        types = [Vars.Resid.OWNERSHIP, Vars.Resid.RESID_TYPE]
        data = (
            self.lf.filter(pl.col(Vars.CONSTR).list.len() != 0)
            .group_by(*types, Vars.CONSTR)
            .len('count')
            .sort([*types, 'count'], descending=[False, False, True])
            .with_columns(
                pl.col(Vars.CONSTR).list.join(', '),
                total=pl.sum('count').over(types),
            )
            .with_columns(ratio=pl.col('count') / pl.col('total'))
            .collect()
        )

        output = self.conf.dirs.analysis / '0101.주거 시공 비율'
        output.mkdir(exist_ok=True)

        data.write_excel(output.parent / f'{output.name}.xlsx', column_widths=150)
        self._plot_const_ratio(data=data, types=types, output=output)

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

    def _anova_data(self, ownership: str, resid_type: str):
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
        import pingouin as pg  # noqa: PLC0415

        lf, between = self._anova_data(ownership, resid_type)

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
def regional_cost(*, conf: Config):
    """지역별 지원액."""
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
def residential(*, conf: Config):
    """가구 유형별 분석."""
    r = _ResidentialTypes(conf)

    r.anova()
    r.plot_ownership()
    r.plot_types_count()
    r.plot_const_ratio()

    for e, b in itertools.product([True, False], [True, False]):
        r.plot_cost(etc=e, boiler=b)


@app.command
def residential_cost(
    *,
    scale=0.6,
    pad=45,
    max_intersection_count: int = 10,
    conf: Config,
):
    """시공 유형별 가구 관급자재, 시공업체계약금액."""
    utils.mpl.MplTheme(scale).grid().apply()
    material = pl.col(Vars.COST) - pl.col(Vars.COST_CONTRACTOR)
    data = (
        pl.scan_parquet(conf.source(BldgType.RESIDENTIAL))
        .drop_nulls(f'{Vars.CONSTR}(원본)')
        .filter(pl.col(Vars.CONSTR).list.len() != 0)
        .select(
            pl.col(Vars.CONSTR).list.join(', '),
            pl.col(Vars.COST_CONTRACTOR) / 10000,
            material.truediv(10000).alias(Vars.COST_MATERIAL),
        )
        .with_columns(pl.len().over(Vars.CONSTR).alias('count'))
        .with_columns(
            pl.col('count').rank('dense', descending=True).alias('rank'),
            (pl.col('count') / pl.len()).alias('ratio'),
        )
        .filter(pl.col('rank') <= max_intersection_count)
        .sort('count', descending=True)
        .collect()
    )

    const_order = data[Vars.CONSTR].unique(maintain_order=True)
    counts = data.unique(Vars.CONSTR, maintain_order=True)

    axes: tuple[Axes, Axes]
    fig, axes = plt.subplots(1, 2)

    sns.barplot(counts, x='count', y=Vars.CONSTR, order=const_order, ax=axes[0])

    axes[0].invert_xaxis()
    axes[0].bar_label(
        axes[0].containers[0],  # type: ignore[arg-type]
        labels=[f'{x:.1%}' for x in counts['ratio']],
        label_type='edge',
        padding=2,
        color='.25',
    )
    sns.violinplot(
        data.unpivot(
            [Vars.COST_CONTRACTOR, Vars.COST_MATERIAL], index=Vars.CONSTR
        ).with_columns(pl.col('variable').str.strip_suffix('액')),
        x='value',
        y=Vars.CONSTR,
        order=const_order,
        hue='variable',
        ax=axes[1],
        linewidth=0.5,
        density_norm='width',
    )

    axes[0].set_yticklabels([])
    axes[0].set_xlabel('건수')
    axes[1].set_xlabel('비용 (만원)')
    axes[1].get_legend().set_title('')

    axes[1].yaxis.set_tick_params(pad=pad)
    for tick in axes[1].yaxis.get_majorticklabels():
        tick.set_horizontalalignment('center')

    axes[0].margins(x=0.2, y=0.01)
    axes[1].margins(y=0.01)

    for ax in axes:
        ax.set_ylabel('')
        ax.autoscale_view()

    fig.savefig(conf.dirs.analysis / '0102.주거 자재-시공업체 금액.png')


@app.command
def social(*, max_const: int = 8, conf: Config):
    """사회복지시설 유형별 분석."""
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
        .filter(pl.col('rank') <= max_const)
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
