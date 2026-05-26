from __future__ import annotations

import tempfile
import unittest
import os

from silica.kernel.ops import Op, OpType
from silica.kernel.ops_io import load_ops, dump_ops


class TestOpsIO(unittest.TestCase):
    def test_ops_io_round_trip(self):
        # Create some ops
        ops = [
            Op(
                op=OpType.write,
                heading="Concept A",
                source_basename="inbox.md",
                path="notes/Concept A.md",
                snippet="Snippet A",
                hub="HubNote",
            ),
            Op(
                op=OpType.skip,
                heading="Concept B",
                source_basename="inbox.md",
                reason="duplicate",
            ),
        ]

        # Use temporary file
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        try:
            # Dump ops
            dump_ops(path, ops)

            # Load ops
            loaded = load_ops(path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].op, OpType.write)
            self.assertEqual(loaded[0].heading, "Concept A")
            self.assertEqual(loaded[0].path, "notes/Concept A.md")
            self.assertEqual(loaded[0].snippet, "Snippet A")
            self.assertEqual(loaded[0].hub, "HubNote")

            self.assertEqual(loaded[1].op, OpType.skip)
            self.assertEqual(loaded[1].heading, "Concept B")
            self.assertEqual(loaded[1].reason, "duplicate")
            self.assertIsNone(loaded[1].path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
