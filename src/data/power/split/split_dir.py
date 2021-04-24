"""
The `Split Directory` contains the files that define the triples
split into train/valid/test.

**Structure**

::

    split/

        entities.tsv
        relations.tsv

        train_entities.tsv
        valid_entities.tsv
        test_entities.tsv

        train_facts.tsv
        
        valid_facts_known.tsv
        valid_facts_unknown.tsv

        test_facts_known.tsv
        test_facts_unknown.tsv

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.power.split.facts_tsv import FactsTsv
from data.power.split.labels_tsv import LabelsTsv


class SplitDir(BaseDir):
    entities_tsv: LabelsTsv
    relations_tsv: LabelsTsv

    train_entities_tsv: LabelsTsv
    valid_entities_tsv: LabelsTsv
    test_entities_tsv: LabelsTsv

    train_facts_tsv: FactsTsv

    valid_facts_known_tsv: FactsTsv
    valid_facts_unknown_tsv: FactsTsv

    test_facts_known_tsv: FactsTsv
    test_facts_unknown_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.entities_tsv = LabelsTsv(path.joinpath('entities.tsv'))
        self.relations_tsv = LabelsTsv(path.joinpath('relations.tsv'))

        self.train_entities_tsv = LabelsTsv(path.joinpath('train_entities.tsv'))
        self.valid_entities_tsv = LabelsTsv(path.joinpath('valid_entities.tsv'))
        self.test_entities_tsv = LabelsTsv(path.joinpath('test_entities.tsv'))

        self.train_facts_tsv = FactsTsv(path.joinpath('train_facts.tsv'))

        self.valid_facts_known_tsv = FactsTsv(path.joinpath('valid_facts_known.tsv'))
        self.valid_facts_unknown_tsv = FactsTsv(path.joinpath('valid_facts_unknown.tsv'))

        self.test_facts_known_tsv = FactsTsv(path.joinpath('test_facts_known.tsv'))
        self.test_facts_unknown_tsv = FactsTsv(path.joinpath('test_facts_unknown.tsv'))

    def check(self) -> None:
        super().check()

        self.entities_tsv.check()
        self.relations_tsv.check()

        self.train_entities_tsv.check()
        self.valid_entities_tsv.check()
        self.test_entities_tsv.check()

        self.train_facts_tsv.check()

        self.valid_facts_known_tsv.check()
        self.valid_facts_unknown_tsv.check()

        self.test_facts_known_tsv.check()
        self.test_facts_unknown_tsv.check()
