"""Tests for targeted synthetic data generation."""

from __future__ import annotations

import json
import unittest

from training.generate_targeted_data import (
    make_targeted_pretrain_rows,
    make_targeted_sft_rows,
)


class TargetedDataTests(unittest.TestCase):
    def test_targeted_sft_rows_are_compact_json_tool_lists(self) -> None:
        rows = make_targeted_sft_rows()
        self.assertGreaterEqual(len(rows), 16)
        behaviors = {row["behavior"] for row in rows}
        self.assertIn("red_herring_rejection", behaviors)
        self.assertIn("cascading_origin_trace", behaviors)
        self.assertIn("communicator_uncertainty", behaviors)

        for row in rows:
            content = row["messages"][-1]["content"]
            actions = json.loads(content)
            self.assertIsInstance(actions, list)
            self.assertLessEqual(len(actions), 9)
            self.assertEqual(actions[-1]["tool_name"], "finish_incident")
            self.assertNotIn("```", content)

    def test_targeted_pretrain_rows_name_specific_failure_modes(self) -> None:
        rows = make_targeted_pretrain_rows()
        text = "\n".join(row["text"].lower() for row in rows)
        self.assertIn("red herring", text)
        self.assertIn("cascading chain", text)
        self.assertIn("compact json array", text)
        self.assertIn("communicator", text)


if __name__ == "__main__":
    unittest.main()
