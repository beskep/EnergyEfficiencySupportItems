# ruff: noqa: PLC0415

from __future__ import annotations

import dataclasses as dc
import enum
import itertools
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import numpy as np
import pandas as pd
import polars as pl
import rich
from loguru import logger
from scipy import optimize

from eesi import utils
from eesi.config import BldgType, Config, Vars
from eesi.s01prep import Preprocess
from eesi.utils.terminal import Progress

if TYPE_CHECKING:
    from collections.abc import Iterable


class Cost(enum.StrEnum):
    TOTAL = Vars.COST
    CONTRACTOR = Vars.COST_CONTRACTOR
    MATERIAL = Vars.COST_MATERIAL


app = utils.cli.App(
    config=cyclopts.config.Toml('config.toml', use_commands_as_keys=False)
)


@app.command
def bayesian_test(_conf: Config):
    import arviz as az
    import pymc
    import pymc.math

    # 참값
    a = 1.0
    b = 2.0
    c = 3.0

    rng = np.random.default_rng(42)
    m = rng.choice([0, 1], size=(100, 3))

    y = np.matmul(m, np.array([a, b, c])) + rng.normal(0, 0.01, size=m.shape[0])

    with pymc.Model():
        x = pymc.HalfNormal('x', shape=3)
        sigma = pymc.HalfNormal('sigma')
        mu = pymc.math.dot(m, x)

        pymc.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pymc.sample(10, tune=10, target_accept=0.95)

    summary = az.summary(trace)
    rich.print(summary)


@dc.dataclass
class _CostEstimator:
    conf: Config

    FAN: ClassVar[str] = '선풍기'

    construction: str = f'{Vars.CONSTR}(원본)'
    constructions: tuple[str, ...] = tuple(
        sorted({FAN, *Preprocess.CONSTRUCTION.values()})
    )

    @dc.dataclass
    class Matrix:
        bldg: BldgType
        cost: Cost

        m: np.ndarray
        total_cost: np.ndarray
        # total_cost == m @ cost  # noqa: ERA001

    @dc.dataclass
    class Case:
        bldg: BldgType
        cost: Cost
        exclude_fan: bool

        @staticmethod
        def _iter():
            for bldg, cost, ef in itertools.product(BldgType, Cost, [True, False]):
                if bldg is BldgType.SOCIAL_SERVICE and cost is not Cost.TOTAL:
                    continue

                yield bldg, cost, ef

        @classmethod
        def iter(cls, *, track: bool = True) -> Iterable[tuple[BldgType, Cost, bool]]:
            it = cls._iter()
            if track:
                it = Progress.iter(tuple(it))
            return it

    def matrix(
        self,
        bldg: BldgType,
        cost: Cost,
        *,
        samples: int | None = None,
        exclude_fan: bool = True,
    ) -> Matrix:
        lf = (
            pl.scan_parquet(self.conf.source(bldg))
            .select(self.construction, cost)
            .drop_nulls(self.construction)
        )

        const = pl.col(self.construction)

        if exclude_fan:
            lf = lf.filter(const.list.contains(self.FAN).not_())

        for c in self.constructions:
            lf = lf.with_columns(const.list.contains(c).cast(pl.Int8).alias(c))

        df = lf.collect()

        if samples and df.height > samples:
            df = df.sample(samples, shuffle=True, seed=42)

        m = df.select(self.constructions).to_numpy()
        cost_array = df.select(cost).to_numpy() / 10000  # 만원 단위

        return self.Matrix(bldg=bldg, cost=cost, m=m, total_cost=cost_array)

    def lstsq(self, bldg: BldgType, cost: Cost, *, exclude_fan: bool = True):
        matrix = self.matrix(bldg, cost, exclude_fan=exclude_fan)
        costs, _residuals, _rank, _s = np.linalg.lstsq(
            matrix.m, matrix.total_cost.ravel()
        )
        return costs

    def nnls(self, bldg: BldgType, cost: Cost, *, exclude_fan: bool = True):
        matrix = self.matrix(bldg, cost, exclude_fan=exclude_fan)
        return optimize.nnls(matrix.m, matrix.total_cost.ravel())

    def bayesian(
        self,
        bldg: BldgType,
        cost: Cost,
        *,
        draws: int = 10,
        samples: int | None = 100,
        exclude_fan: bool = True,
    ):
        import arviz as az
        import pymc
        import pymc.math

        matrix = self.matrix(bldg, cost, samples=samples, exclude_fan=exclude_fan)

        with pymc.Model():
            costs = pymc.Uniform(
                'costs', lower=0, upper=200, shape=len(self.constructions)
            )
            sigma = pymc.HalfNormal('sigma', sigma=10)
            mu = pymc.math.dot(matrix.m, costs)
            pymc.Normal('total_cost', mu=mu, sigma=sigma, observed=matrix.total_cost)
            trace = pymc.sample(draws, tune=int(draws / 2), random_seed=42)

        summary = az.summary(trace)
        assert isinstance(summary, pd.DataFrame)
        return pl.from_pandas(summary.reset_index()).select(
            pl.Series('construction', [*self.constructions, 'sigma']), pl.all()
        )

    def _lstsq(self):
        def it():
            for bldg, cost, exclude_fan in self.Case.iter():
                logger.info('{}|{}|{}', bldg, cost, exclude_fan)
                costs = self.lstsq(bldg, cost, exclude_fan=exclude_fan)

                yield pl.DataFrame({
                    'building': bldg,
                    'cost': cost,
                    'exclude-fan': exclude_fan,
                    'construction': self.constructions,
                    'costs': costs,
                })

        return pl.concat(it())

    def _nnls(self):
        def it():
            for bldg, cost, exclude_fan in self.Case.iter():
                logger.info('{}|{}|{}', bldg, cost, exclude_fan)
                costs, residual = self.nnls(bldg, cost, exclude_fan=exclude_fan)

                yield pl.DataFrame({
                    'building': bldg,
                    'cost': cost,
                    'exclude-fan': exclude_fan,
                    'construction': self.constructions,
                    'costs': costs,
                    'residual': residual,
                })

        return pl.concat(it())

    def _bayesian(self):
        console = rich.get_console()
        for bldg, cost, exclude_fan in self.Case.iter(track=False):
            console.print(bldg, cost, exclude_fan)
            console.print(self.bayesian(bldg, cost, exclude_fan=exclude_fan))

    def __call__(self, method: Literal['lstsq', 'nnls', 'bayesian'], **kwargs):
        match method:
            case 'lstsq':
                return self._lstsq()
            case 'nnls':
                return self._nnls()
            case 'bayesian':
                pl.Config.set_tbl_rows(20)
                return self._bayesian(**kwargs)

        raise ValueError(method)


@app.command
def estimate(
    method: Literal['lstsq', 'nnls', 'bayesian'] = 'nnls',
    *,
    conf: Config,
):
    """각 시공 가격 추정."""
    estimator = _CostEstimator(conf=conf)
    data = estimator(method)

    if data is not None:
        d = conf.dirs.cost
        d.mkdir(exist_ok=True)
        data.write_excel(d / f'비용 추정-{method}.xlsx', column_widths=120)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()
    app()
