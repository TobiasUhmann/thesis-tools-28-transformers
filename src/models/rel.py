from dataclasses import dataclass


@dataclass(frozen=True)
class Rel:
    id: int
    lbl: str
