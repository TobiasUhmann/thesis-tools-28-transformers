"""
The `POWER Classes TSV` gives detailed information about the classes used in
the `POWER Samples TSV`s.
"""

import csv
from pathlib import Path
from typing import List, Tuple

from data.base_file import BaseFile


class ClassesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self) -> List[Tuple[int, int, float, str]]:
        """
        :return: [(rel, tail, freq, label)]
        """

        with open(self.path, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader)

            rows = []
            for rel, tail, freq, label in csv_reader:
                rows.append((int(rel), int(tail), float(freq), label))

        return rows
