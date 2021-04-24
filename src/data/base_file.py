import errno
import os
from os.path import isfile
from pathlib import Path


class BaseFile:
    path: Path

    def __init__(self, path: Path):
        self.path = path

    def check(self, should_exist=True) -> None:
        if should_exist:
            if not isfile(self.path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)
        else:
            if isfile(self.path):
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), self.path)
