"""
The `Ryn Split Directory` contains the files that define the triples
split into train/valid/test.

**Structure**

::

    split/                 # Ryn Split Directory

        entity2id.txt      # Ryn Entity Labels TXT
        relation2id.txt    # Ryn Relation Labels TXT

        cw.train2id.txt    # Ryn CW Train Triples TXT
        cw.valid2id.txt    # Ryn CW Valid Triples TXT
        ow.valid2id.txt    # Ryn OW Valid Triples TXT
        ow.test2id.txt     # Ryn OW Test Triples TXT

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.split.labels_txt import LabelsTxt
from dao.ryn.split.triples_txt import TriplesTxt


class SplitDir(BaseDir):

    ent_labels_txt: LabelsTxt
    rel_labels_txt: LabelsTxt
    
    cw_train_triples_txt: TriplesTxt
    cw_valid_triples_txt: TriplesTxt
    ow_valid_triples_txt: TriplesTxt
    ow_test_triples_txt: TriplesTxt

    def __init__(self, path: Path):
        super().__init__(path)

        self.ent_labels_txt = LabelsTxt(path.joinpath('entity2id.txt'))
        self.rel_labels_txt = LabelsTxt(path.joinpath('relation2id.txt'))
        
        self.cw_train_triples_txt = TriplesTxt(path.joinpath('cw.train2id.txt'))
        self.cw_valid_triples_txt = TriplesTxt(path.joinpath('cw.valid2id.txt'))
        self.ow_valid_triples_txt = TriplesTxt(path.joinpath('ow.valid2id.txt'))
        self.ow_test_triples_txt = TriplesTxt(path.joinpath('ow.test2id.txt'))

    def check(self) -> None:
        super().check()
            
        self.ent_labels_txt.check()
        self.rel_labels_txt.check()

        self.cw_train_triples_txt.check()
        self.cw_valid_triples_txt.check()
        self.ow_valid_triples_txt.check()
        self.ow_test_triples_txt.check()
