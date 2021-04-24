from dataclasses import dataclass
from typing import List, Tuple

from models.fact import Fact
from models.rule import Rule


@dataclass(frozen=True)
class Pred:
    fact: Fact
    conf: float
    sents: List[Tuple[str, float]]
    rules: List[Rule]

