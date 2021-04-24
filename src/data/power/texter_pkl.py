import pickle
from pathlib import Path

from data.base_file import BaseFile
from power.texter import Texter


class TexterPkl(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, texter: Texter) -> None:
        with open(self.path, 'wb') as f:
            pickle.dump(texter, f)

    def load(self) -> Texter:
        with open(self.path, 'rb') as f:
            return pickle.load(f)
