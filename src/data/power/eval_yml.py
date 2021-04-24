"""
The `Power Eval Yml` stores the results from evaluating `Power` components.
"""

from pathlib import Path
from typing import Any

import yaml

from data.base_file import BaseFile


class EvalYml(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, data: Any) -> None:
        with open(self.path, 'w', encoding='utf8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def load(self) -> Any:
        with open(self.path, 'r', encoding='utf8') as f:
            return yaml.safe_load(f)
