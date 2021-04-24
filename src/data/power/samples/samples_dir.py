"""
The `Power Samples Directory` contains the input files required for training the
`Power Classifier`. The `Power Temp Directory` keeps intermediate files
for debugging purposes.

**Structure**

::

    power/                 # Power Samples Directory

        tmp/               # Power Temp Directory

        classes.tsv        # Power Classes TSV

        test_samples.tsv   # Power Test Samples TSV
        train_samples.tsv  # Power Train Samples TSV
        valid_samples.tsv  # Power Valid Samples TSV

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.power.samples.classes_tsv import ClassesTsv
from data.power.samples.samples_tsv import SamplesTsv


class SamplesDir(BaseDir):
    classes_tsv: ClassesTsv

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.classes_tsv = ClassesTsv(path.joinpath('classes.tsv'))

        self.train_samples_tsv = SamplesTsv(path.joinpath('train.tsv'))
        self.valid_samples_tsv = SamplesTsv(path.joinpath('valid.tsv'))
        self.test_samples_tsv = SamplesTsv(path.joinpath('test.tsv'))

    def check(self) -> None:
        super().check()

        self.classes_tsv.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()
