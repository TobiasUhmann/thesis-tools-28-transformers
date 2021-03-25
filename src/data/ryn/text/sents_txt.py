"""
The `Ryn Sentences TXT` stores the entities' sentences.

* Header row
* Entity RID | Entity label | Sentences

**Example**

::

    # Format: <ID> | <NAME> | <SENTENCE>
    0 | Dominican Republic | Border disputes under Trujillo ...
    0 | Dominican Republic | In 2006, processed shells were ...

|
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

from data.base_file import BaseFile


class SentsTxt(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self) -> Dict[int, Set[str]]:
        """
        :return: {entity RID: {entity sentences}}
        """

        # Read all lines into memory
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        # Assert that first line is doc header
        assert lines[0].startswith('#')

        #
        # Parse doc body
        #
        # Each line should have the format
        # entity RID | entity label | sentence
        #

        ent_to_sents: Dict[int, Set[str]] = defaultdict(set)

        for line in lines[1:]:
            ent, _, sent = line.split(' | ')
            ent_to_sents[int(ent)].add(sent.strip())

        return ent_to_sents
