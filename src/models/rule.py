from dataclasses import dataclass
from typing import List

from models.fact import Fact


@dataclass(frozen=True)
class Rule:
    fires: int
    holds: int
    conf: float
    head: Fact
    body: List[Fact]

    def __repr__(self):
        return f"[fires={self.fires}, holds={self.holds}, conf={self.conf:.2f}," \
               f" {self.head} <= {', '.join([str(fact) for fact in self.body])}]"
