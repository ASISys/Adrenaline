import glob
import statistics

from dataclasses import dataclass

from matplotlib import pyplot
from evaluation.plots.dataset_performance.parse_performance_metrics import (
    get_ttft_for_path,
    get_tpot_for_path,
    get_thpt_for_path,
    get_p99_tpot_for_path,
)

@dataclass
class PlotConfig:
    figsize = (2.5, 2)
    colors = ['#0000ED', '#BF0000']
    marker_style = ['o', 's', '^', 'v', 'D', '<', '>', 's', 'x', '*']
    markersize = 6
    markeredgewidth = 1.5
    grid_linewidth = 0.3


def get_ttft_for_config(config: str, result_dir: str, num_execs: int):
    result_paths = glob.glob(f"{result_dir}/{config}*")
    assert len(result_paths) > 0, f"{result_dir}/{config} not exist evaluation results"
    result_paths = result_paths[:num_execs]
    ttfts = [
        get_ttft_for_path(f"{result_path}/output.txt") for result_path in result_paths
    ]
    avg_ttft = statistics.fmean(ttfts)
    return avg_ttft


def get_tpot_for_config(config: str, result_dir: str, num_execs: int):
    result_paths = glob.glob(f"{result_dir}/{config}*")
    assert len(result_paths) > 0, f"{config} not exist evaluation results"
    result_paths = result_paths[:num_execs]
    tpots = [
        get_tpot_for_path(f"{result_path}/output.txt") for result_path in result_paths
    ]
    avg_tpot = statistics.fmean(tpots)
    return avg_tpot


def get_p99_tpot_for_config(config: str, result_dir: str, num_execs: int):
    result_paths = glob.glob(f"{result_dir}/{config}*")
    assert len(result_paths) > 0, f"{config} not exist evaluation results"
    result_paths = result_paths[:num_execs]
    p99_tpots = [
        get_p99_tpot_for_path(f"{result_path}/output.txt") for result_path in result_paths
    ]
    p99_tpot = statistics.fmean(p99_tpots)
    return p99_tpot


def get_thpt_for_config(config: str, result_dir: str, num_execs: int, result_name: str = "decode.out"):
    result_paths = glob.glob(f"{result_dir}/{config}*")
    assert len(result_paths) > 0, f"{config} not exist evaluation results"
    result_paths = result_paths[:num_execs]
    thpts = [
        get_thpt_for_path(f"{result_path}/{result_name}") for result_path in result_paths
    ]
    avg_thpt = statistics.fmean(thpts)
    return avg_thpt


def plot_ttft(
    qps_list: list[float],
    offload_ttfts: dict[float, list[float]],
    disagg_ttfts: list[float],
    output_path: str,
):
    for offload_ratio, offload_ttft in offload_ttfts.items():
        pyplot.plot(
            qps_list, offload_ttft, "-o", label=f"offload (ratio={offload_ratio})"
        )
    pyplot.plot(qps_list, disagg_ttfts, "--o", label="disagg")
    pyplot.xlabel("QPS")
    pyplot.ylabel("TTFT (ms)")
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.savefig(output_path)
    pyplot.clf()


def plot_tpot(
    qps_list: list[float],
    offload_tpots: dict[float, list[float]],
    disagg_tpots: list[float],
    output_path: str,
):
    for offload_ratio, offload_tpot in offload_tpots.items():
        pyplot.plot(
            qps_list, offload_tpot, "-o", label=f"offload (ratio={offload_ratio})"
        )
    pyplot.plot(qps_list, disagg_tpots, "--o", label="disagg")
    pyplot.xlabel("QPS")
    pyplot.ylabel("TPOT (ms)")
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.savefig(output_path)
    pyplot.clf()


def plot_stable_thpt(
    qps_list: list[float],
    offload_thpts: dict[float, list[float]],
    disagg_thpts: list[float],
    output_path: str,
):
    for offload_ratio, offload_thpt in offload_thpts.items():
        pyplot.plot(
            qps_list, offload_thpt, "-o", label=f"offload (ratio={offload_ratio})"
        )
    pyplot.plot(qps_list, disagg_thpts, "--o", label="disagg")
    pyplot.xlabel("QPS")
    pyplot.ylabel("Stable Throughput (tokens/ms)")
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.savefig(output_path)
    pyplot.clf()
