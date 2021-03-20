"""
The `Ryn Text Directory` contains the entities' sentences.

**Structure**

::

    text/                         # Ryn Text Directory

        cw.train-sentences.txt    # Ryn CW Train Sentences TXT
        ow.valid-sentences.txt    # Ryn OW Valid Sentences TXT
        ow.test-sentences.txt     # Ryn OW Test Sentences TXT

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.text.sents_txt import SentsTxt


class TextDir(BaseDir):

    cw_train_sents_txt: SentsTxt
    ow_valid_sents_txt: SentsTxt
    ow_test_sents_txt: SentsTxt

    def __init__(self, path: Path):
        super().__init__(path)
        
        self.cw_train_sents_txt = SentsTxt(path.joinpath('cw.train-sentences.txt'))
        self.ow_valid_sents_txt = SentsTxt(path.joinpath('ow.valid-sentences.txt'))
        self.ow_test_sents_txt = SentsTxt(path.joinpath('ow.test-sentences.txt'))

    def check(self) -> None:
        super().check()

        self.cw_train_sents_txt.check()
        self.ow_valid_sents_txt.check()
        self.ow_test_sents_txt.check()
