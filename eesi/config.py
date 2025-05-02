from __future__ import annotations

import dataclasses as dc
import enum
from pathlib import Path

import cyclopts


class BldgType(enum.StrEnum):
    RESIDENTIAL = '가구'
    SOCIAL_SERVICE = '사회복지시설'

    R = RESIDENTIAL
    S = SOCIAL_SERVICE


class Vars:
    YEAR: str = '사업연도'
    KEY: str = '기준키'
    CONSTR: str = '시공내역'
    COST: str = '총지원금액'
    CONTRACTOR_PAYMENTS: str = '시공업체계약금액'

    class Residual:
        # 가구
        REGISTRATION_DATE: str = '등록일'
        SUPPORT_TYPE: str = '보호유형'
        RESID_TYPE: str = '주택유형'
        OWNERSHIP: str = '주거실태'  # 자가, 전세, 월세, ...

    class SocialService:
        # 사회복지시설
        EXISTING_BOILER: str = '기존보일러유무'
        EXISTING_BOILER_FUEL: str = '기존보일러연료'
        EXISTING_BOILER_FUEL_COLS: tuple[str, ...] = (
            '기름일반',
            '기름고효율',
            '가스일반',
            '가스콘덴싱',
        )

    Resid = Residual
    Social = SocialService


@dc.dataclass
class Dirs:
    raw: Path = Path('0000.raw')
    data: Path = Path('0001.data')
    analysis: Path = Path('0100.analysis')


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config:
    root: Path
    dirs: Dirs = dc.field(default_factory=Dirs)

    def __post_init__(self):
        self.update()

    def update(self):
        for field in (f.name for f in dc.fields(self.dirs)):
            v = getattr(self.dirs, field)
            setattr(self.dirs, field, self.root / v)

        return self

    def source(self, bldg: BldgType, /):
        return self.dirs.data / f'0001.{bldg}.parquet'


if __name__ == '__main__':
    import cyclopts
    import rich

    conf = cyclopts.config.Toml('config.toml').config
    rich.print(conf)
