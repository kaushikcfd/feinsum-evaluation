import dataclasses as dc
from pytools.tag import UniqueTag


@dc.dataclass(frozen=True)
class NamedAxis(UniqueTag):
    name: str
