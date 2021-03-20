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
from typing import List, Tuple

from torchtext.data import TabularDataset, Field
from torchtext.vocab import Vocab

from dao.base_dir import BaseDir
from dao.ower.classes_tsv import ClassesTsv
from dao.ower.samples_tsv import SamplesTsv
from dao.ower.tmp.tmp_dir import TmpDir
from dao.ryn.split.labels_txt import LabelsTxt


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

    def read_datasets(self, class_count: int, sent_count: int, vectors=None) \
            -> Tuple[List[Sample], List[Sample], List[Sample], Vocab]:
        """
        :param vectors: Pre-trained word embeddings
        """

        def tokenize(text: str) -> List[str]:
            return text.split()

        ent_field = Field(sequential=False, use_vocab=False)
        ent_label_field = Field()
        class_field = Field(sequential=False, use_vocab=False)
        sent_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)

        ent_col = ('ent', ent_field)
        ent_label_col = ('ent_label', ent_label_field)
        class_cols = [(f'class_{i}', class_field) for i in range(class_count)]
        sent_cols = [(f'sent_{i}', sent_field) for i in range(sent_count)]

        cols = [ent_col, ent_label_col] + class_cols + sent_cols

        train_tab_set = TabularDataset(str(self.train_samples_tsv.path), 'tsv', cols, skip_header=True)
        valid_tab_set = TabularDataset(str(self.valid_samples_tsv.path), 'tsv', cols, skip_header=True)
        test_tab_set = TabularDataset(str(self.test_samples_tsv.path), 'tsv', cols, skip_header=True)

        #
        # Build vocab on train data
        #

        sent_field.build_vocab(train_tab_set, vectors=vectors)
        vocab = sent_field.vocab

        #
        # Transform TabularDataset -> List[Sample]
        #

        def transform(raw_set: TabularDataset) -> List[Sample]:
            return [Sample(
                int(getattr(row, 'ent')),
                [int(getattr(row, f'class_{i}')) for i in range(class_count)],
                [[vocab[token] for token in getattr(row, f'sent_{i}')] for i in range(sent_count)]
            ) for row in raw_set]

        train_set = transform(train_tab_set)
        valid_set = transform(valid_tab_set)
        test_set = transform(test_tab_set)

        return train_set, valid_set, test_set, vocab
