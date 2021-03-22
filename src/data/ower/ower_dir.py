"""
The `OWER Directory` contains the input files required for training the
`OWER Classifier`. The `OWER Temp Directory` keeps intermediate files
for debugging purposes.

**Structure**

::

    ower/                 # OWER Directory

        tmp/              # OWER Temp Directory

        ent_labels.txt    # OWER Entity Labels TXT
        rel_labels.txt    # OWER Relation Labels TXT

        classes.tsv       # OWER Classes TSV

        test.tsv          # OWER Test Samples TSV
        train.tsv         # OWER Train Samples TSV
        valid.tsv         # OWER Valid Samples TSV

|
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from data.base_dir import BaseDir
from data.ower.classes_tsv import ClassesTsv
from data.ower.samples_tsv import SamplesTsv
from data.ower.tmp.tmp_dir import TmpDir
from data.ryn.split.labels_txt import LabelsTxt


@dataclass
class Sample:
    ent: int
    classes: List[int]
    sents: List[List[int]]

    def __iter__(self):
        return iter((self.ent, self.classes, self.sents))


class OwerDir(BaseDir):
    tmp_dir: TmpDir

    ent_labels_txt: LabelsTxt
    rel_labels_txt: LabelsTxt

    classes_tsv: ClassesTsv

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.tmp_dir = TmpDir(path.joinpath('tmp'))

        self.ent_labels_txt = LabelsTxt(path.joinpath('ent_labels.txt'))
        self.rel_labels_txt = LabelsTxt(path.joinpath('rel_labels.txt'))

        self.classes_tsv = ClassesTsv(path.joinpath('classes.tsv'))

        self.train_samples_tsv = SamplesTsv(path.joinpath('train.tsv'))
        self.valid_samples_tsv = SamplesTsv(path.joinpath('valid.tsv'))
        self.test_samples_tsv = SamplesTsv(path.joinpath('test.tsv'))

    def check(self) -> None:
        super().check()

        self.tmp_dir.check()

        self.ent_labels_txt.check()
        self.rel_labels_txt.check()

        self.classes_tsv.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()

    def create(self) -> None:
        super().create()

        self.tmp_dir.create()
