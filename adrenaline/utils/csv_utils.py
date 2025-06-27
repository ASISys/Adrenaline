import csv
from typing import Any


def save_to_csv(
    items: list[dict[str, Any]],
    headers: list[str],
    output_path: str,
):
    with open(output_path, "w") as f:
        w = csv.DictWriter(f, headers)
        w.writeheader()
        [w.writerow(item) for item in items]
