"""
The `IRT Labels TXT` contains the entities' or relations' labels.

* Header row specifies number of entities/relations
* Space separated

**Example**

::

    3
    Dominican Republic 0
    republic 1
    Mighty Morphin Power Rangers 2

|
"""

import logging
from pathlib import Path
from typing import Dict

from data.base_file import BaseFile


class LabelsTxt(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self) -> Dict[int, str]:
        """
        :return: {ent/rel RID: ent/rel label}
        """

        # Read all lines into memory
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        # Parse declared entity count from doc header
        declared_ent_count = int(lines[0])

        ## Parse doc body
        ##
        ## Each line should specify the entity followed by its RID
        ## Example: 'Ent label with spaces 123'

        rid_to_label: Dict[int, str] = {}

        for line in lines[1:]:
            parts = line.split()

            # Warn if line.split() != line.split(' ') as non-space whitespace will be lost
            parts_by_space = line.split(' ')
            if len(parts) != len(parts_by_space):
                logging.warning('Line must contain single spaces only as separator.'
                                f' Replacing each whitespace with single space. Line: {repr(line)}')

            label = ' '.join(parts[:-1])
            rid = int(parts[-1])

            rid_to_label[rid] = label

        assert len(rid_to_label) == declared_ent_count

        return rid_to_label
