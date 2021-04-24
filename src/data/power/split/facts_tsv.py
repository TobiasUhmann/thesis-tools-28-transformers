"""
The `Power Facts TSVs` stores triples using RIDs.

**Example**

::

    10195 7677 22
    4253 450 69
    5806 2942 32
    2271 6322 203

|
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from data.base_file import BaseFile


@dataclass(frozen=True)
class Fact:
    head: int
    head_lbl: str

    rel: int
    rel_lbl: str

    tail: int
    tail_lbl: str

    def __iter__(self):
        return iter((self.head, self.head_lbl, self.rel, self.rel_lbl, self.tail, self.tail_lbl))


class FactsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self) -> List[Fact]:
        with open(self.path, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader)

            facts = [Fact(int(head), head_lbl, int(rel), rel_lbl, int(tail), tail_lbl)
                     for head, head_lbl, rel, rel_lbl, tail, tail_lbl in csv_reader]

        return facts
