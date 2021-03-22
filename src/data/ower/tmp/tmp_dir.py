"""
The `OWER Temp Directory` keeps intermediate files for debuggin purposes.

**Structure**

::

    tmp/            # OWER Temp Directory

        train.db    # OWER Train Triples DB
        valid.db    # OWER Valid Triples DB
        test.db     # OWER Test Triples DB

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.ower.tmp.triples_db import TriplesDb


class TmpDir(BaseDir):
    train_triples_db: TriplesDb
    valid_triples_db: TriplesDb
    test_triples_db: TriplesDb

    def __init__(self, path: Path):
        super().__init__(path)

        self.train_triples_db = TriplesDb(path.joinpath('train.db'))
        self.valid_triples_db = TriplesDb(path.joinpath('valid.db'))
        self.test_triples_db = TriplesDb(path.joinpath('test.db'))

    def check(self) -> None:
        super().check()

        self.train_triples_db.check()
        self.valid_triples_db.check()
        self.test_triples_db.check()
