import dataclasses
import statistics

from pathlib import Path
from typing import Optional

SYS_NAME="Adrenaline"

@dataclasses.dataclass
class CommonColors:
    gradient_blues = ['white', '#85D6E7', '#408EC1', '#07428E', '#BF0000']
    gradient_reds = ['white', '#EAAABB', '#D45555', '#BF0000']
    bak_colors = ['#408EC1', '#D45555', 'C2', 'lightgray', '#F56D42']
    vivid_colors = ['black', '#0000ED', '#BF0000', '#006521', '#6500CC']


def calc_mean_with_process(data: list, filter_pct: Optional[float] = 0.1):
    data = sorted(data)
    if filter_pct is not None:
        length = len(data)
        data = data[int(length * filter_pct) : int(length * (1 - filter_pct))]
    mean_data: float = statistics.mean(data)
    return mean_data


def crop_margins(output_path: str, crop_args: str = "-p 10"):
    import pdfCropMargins
    crop_path = Path(output_path)
    crop_path = crop_path.with_stem(crop_path.stem + "_cropped")
    pdfCropMargins.crop(crop_args.split() + f"{output_path} -o {crop_path}".split())
    return crop_path

