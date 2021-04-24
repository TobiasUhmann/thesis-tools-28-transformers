from dataclasses import dataclass


@dataclass(frozen=True)
class Ent:
    id: int
    lbl: str
