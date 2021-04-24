"""
The `POWER Samples TSV` contains the input data for training the
`POWER Classifier`.

* Tabular separated
* 1 Header Row
* First column: Entity RID
* N columns: class_1 .. class_n
* M sentences

**Example**

::

    entity  class_1 class_2 class_3 class_4 sent_1  sent_2  sent_3
    1   0	0	0	0	Foo.    Bar.    Baz.
    2   0	1	0	1	Lorem.  Ypsum.  Dolor

|
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from data.base_file import BaseFile


@dataclass
class Sample:
    ent: int
    ent_label: str
    classes: List[int]
    sents: List[str]

    def __iter__(self):
        return iter((self.ent, self.ent_label, self.classes, self.sents))


class SamplesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self, class_count: int, sent_count: int) -> List[Sample]:
        samples = []

        with open(self.path, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')

            for row in csv_reader:
                samples.append(Sample(
                    int(row[0]),
                    row[1].strip(),
                    [int(c) for c in row[2:2 + class_count]],
                    [s.strip() for s in row[-sent_count:]]
                ))

        return samples
