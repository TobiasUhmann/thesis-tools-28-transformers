"""
The `Ryn Directory` contains the `Ryn Dasaset`, i.e. the triples split
as well as the sentences describing the entities.

**Structure**

::

    ryn/          # Ryn Directory

        split/    # Ryn Split Directory
        text/     # Ryn Text Directory

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.split.split_dir import SplitDir
from dao.ryn.text.text_dir import TextDir


class RynDir(BaseDir):

    split_dir: SplitDir
    text_dir: TextDir

    def __init__(self, path: Path):
        super().__init__(path)

        self.split_dir = SplitDir(path.joinpath('split'))
        self.text_dir = TextDir(path.joinpath('text'))

    def check(self) -> None:
        super().check()

        self.split_dir.check()
        self.text_dir.check()

    def create(self) -> None:
        super().create()

        self.split_dir.create()
        self.text_dir.create()
