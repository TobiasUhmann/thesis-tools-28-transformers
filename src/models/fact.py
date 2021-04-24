from dataclasses import dataclass
from typing import Union, Dict

from models.var import Var

from models.ent import Ent
from models.rel import Rel


@dataclass(frozen=True)
class Fact:
    head: Union[Ent, Var]
    rel: Rel
    tail: Union[Ent, Var]

    @staticmethod
    def from_ints(head: int, rel: int, tail: int, ent_to_lbl: Dict[int, str], rel_to_lbl: Dict[int, str]):
        return Fact(Ent(head, ent_to_lbl[head]),
                    Rel(rel, rel_to_lbl[rel]),
                    Ent(tail, ent_to_lbl[tail]))

    def __repr__(self):
        head_str = f'{self.head.lbl} ({self.head.id})' if type(self.head) == Ent else self.head.name
        tail_str = f'{self.tail.lbl} ({self.tail.id})' if type(self.tail) == Ent else self.tail.name

        return f'({head_str} - {self.rel.lbl} ({self.rel.id}) -> {tail_str})'
