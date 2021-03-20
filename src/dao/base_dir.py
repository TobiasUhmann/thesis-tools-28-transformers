import errno
import os
from os import makedirs
from os.path import isdir
from pathlib import Path


class BaseDir:
    path: Path

    def __init__(self, path: Path):
        self.path = path

    def check(self) -> None:
        if not isdir(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

    def create(self) -> None:
        """
        Create directory if it does not exist already.
        """

        makedirs(self.path, exist_ok=True)
