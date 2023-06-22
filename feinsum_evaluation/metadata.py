import dataclasses as dc


@dc.dataclass(frozen=True)
class NamedAxis:
    name: str
