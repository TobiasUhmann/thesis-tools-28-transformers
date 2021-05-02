"""
The `IRT Text Directory` contains the entities' sentences.

**Structure**

::

    text/                         # IRT Text Directory

        cw.train-sentences.txt    # IRT CW Train Sentences TXT
        ow.valid-sentences.txt    # IRT OW Valid Sentences TXT
        ow.test-sentences.txt     # IRT OW Test Sentences TXT

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.irt.text.sents_txt import SentsTxt


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
