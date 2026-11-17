"""Plain Markdown tables for the generated result files."""
from __future__ import annotations

from typing import Any, List


def markdown_table(header: List[str], rows: List[List[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines
