import csv
import os

from pathlib import Path
from typing import Any


def save_to_csv(
    items: list[dict[str, Any]],
    headers: list[str],
    output_path: str,
):
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        w = csv.DictWriter(f, headers)
        w.writeheader()
        [w.writerow(item) for item in items]
