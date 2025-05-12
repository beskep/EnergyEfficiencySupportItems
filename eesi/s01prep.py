from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cyclopts
import fastexcel
import polars as pl
import polars.selectors as cs
import rich
from loguru import logger

from eesi import utils
from eesi.config import BldgType, Config, Vars

if TYPE_CHECKING:
    from pathlib import Path

app = utils.cli.App(
    config=cyclopts.config.Toml('config.toml', use_commands_as_keys=False)
)


class Preprocess:
    CONSTRUCTION: ClassVar[dict[str, str]] = {
        '가': '가구',
        '곰': '곰팡이',
        '기': '기타',
        '냉': '냉방',
        '단': '단열(벽)',
        '바': '바닥',
        '보': '보일러',
        '진': '진단',
        '창': '창호',
        '천': '단열(천장)',
        '홈': '중문',
        '벽': '단열(벽)',
    }
    CONSTRUCTION_EXCLUDES: tuple[str, ...] = (
        '가구',
        '곰팡이',
        '선풍기',
        '중문',
        '기타',
    )
    OWNERSHIP: ClassVar[dict[str, str]] = {
        # NOTE 전세, 월세, ...는 비자가로 분류
        '자가': '자가',
        '전체무료임차': '전체무료임차',
        '전체무료': '전체무료임차',
        '기타': '기타',
    }

    @staticmethod
    def convert(src: str | bytes | Path):
        reader = fastexcel.read_excel(src)

        def read(sheet: str):
            return (
                reader.load_sheet_by_name(sheet)
                .to_polars()
                .drop('세대주', '생년월일', strict=False)
                .rename(lambda x: x.strip())
                .with_columns(pl.lit(int(sheet)).alias(Vars.YEAR))
            )

        return (
            pl.concat([read(x) for x in reader.sheet_names], how='diagonal_relaxed')
            .select(Vars.YEAR, pl.all().exclude(Vars.YEAR))
            .sort(Vars.YEAR, Vars.KEY)
        )

    @classmethod
    def preprocess(cls, src: str | Path):
        lf = (
            pl.scan_parquet(src)
            .with_columns(
                pl.col(Vars.CONSTR)
                .str.split('|')
                .list.eval(pl.element().replace_strict(cls.CONSTRUCTION))
            )
            .with_columns(
                # 2022년 냉방 -> 선풍기 지원
                pl.when(pl.col(Vars.YEAR) == 2022)  # noqa: PLR2004
                .then(
                    pl.col(Vars.CONSTR)
                    .list.drop_nulls()
                    .list.eval(pl.element().replace('냉방', '선풍기'))
                )
                .otherwise(pl.col(Vars.CONSTR))
                .alias(Vars.CONSTR)
            )
            .with_columns(
                # 시공내역 정렬 ('진단'을 제일 처음으로)
                pl.col(Vars.CONSTR)
                .list.eval(pl.element().replace({'진단': ''}))
                .list.sort()
                .list.eval(pl.element().replace({'': '진단'}))
            )
            .with_columns(
                pl.col(Vars.CONSTR).alias(f'{Vars.CONSTR}(원본)'),
                pl.col(Vars.CONSTR).list.set_difference(cls.CONSTRUCTION_EXCLUDES),
            )
        )

        schema = lf.collect_schema()

        if Vars.Resid.REGISTRATION_DATE in schema:
            lf = lf.with_columns(pl.col(Vars.Resid.REGISTRATION_DATE).dt.date())

        if Vars.COST_CONTRACTOR in schema:
            # 관급자재비용 추가 (총지원금 - 시공업체계약금액)
            cost = pl.col(Vars.COST) - pl.col(Vars.COST_CONTRACTOR)
            lf = lf.with_columns(cost.alias(Vars.COST_MATERIAL))

        if Vars.Resid.OWNERSHIP in schema:
            expr = pl.col(Vars.Resid.OWNERSHIP)
            lf = (
                lf.with_columns(expr.replace({'-': '기타'}))
                .with_columns(
                    expr.alias(f'{Vars.Resid.OWNERSHIP}(원본)'),
                    expr.replace_strict(cls.OWNERSHIP, default='비자가'),
                )
                .with_columns()
            )

        if Vars.Resid.SUPPORT_TYPE in schema:
            lf = lf.with_columns(
                # NOTE 보호유형: 데이터 정제 어려움
                pl.col(Vars.Resid.SUPPORT_TYPE)
                .str.replace_all(' ', '')
                .str.split(',')
                .list.sort()
            )

        if Vars.Social.EXISTING_BOILER in schema:
            lf = lf.with_row_index()
            fuel = (
                lf.select(['index', *Vars.Social.EXISTING_BOILER_FUEL_COLS])
                .unpivot(index='index', variable_name=Vars.Social.EXISTING_BOILER_FUEL)
                .filter(pl.col('value').is_not_null())
                .drop('value')
            )
            lf = (
                lf.drop(Vars.Social.EXISTING_BOILER_FUEL_COLS)
                .select(
                    pl.all().exclude(Vars.Social.EXISTING_BOILER),
                    Vars.Social.EXISTING_BOILER,
                )
                .with_columns(
                    pl.col(Vars.Social.EXISTING_BOILER).replace_strict(
                        {'유': True, '무': False}, return_dtype=pl.Boolean
                    )
                )
                .join(fuel, on='index', how='left')
                .drop('index')
            )

        return lf


@app.command
def convert(*, samples: int = 1000, conf: Config):
    """엑셀 -> parquet 변환."""
    for bldg in BldgType:
        logger.info(bldg)

        src = conf.dirs.raw / f'0000.{bldg}.xlsx'
        data = Preprocess.convert(src)

        data.write_parquet(conf.dirs.data / f'0000.{bldg}.parquet')

        sample = samples < data.height
        data.sample(min(1000, data.height), shuffle=sample).sort(Vars.KEY).write_excel(
            conf.dirs.data / f'0000.{bldg}{"-sample" if sample else ""}.xlsx',
            column_widths=120,
        )
        rich.print(data)


@app.command
def prep(*, samples: int = 1000, conf: Config):
    """데이터 전처리."""
    for bldg in BldgType:
        logger.info(bldg)

        src = conf.dirs.data / f'0000.{bldg}.parquet'
        data = Preprocess.preprocess(src).collect()

        if (error := data.filter(pl.col(Vars.CONSTR).list.contains('ERROR'))).height:
            rich.print(f'error={error}')

        data.write_parquet(conf.dirs.data / f'0001.{bldg}.parquet')

        sample = samples < data.height
        (
            data.with_columns(cs.by_dtype(pl.List).cast(pl.String))
            .sample(min(1000, data.height), shuffle=sample)
            .sort(Vars.KEY)
            .write_excel(
                conf.dirs.data / f'0001.{bldg}{"-sample" if sample else ""}.xlsx',
                column_widths=120,
            )
        )

        rich.print(data)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    app()
