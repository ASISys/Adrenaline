import argparse
import matplotlib

from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from pathlib import Path


from evaluation.plots.dataset_performance.utils import (
    PlotConfig,
    get_ttft_for_config,
    get_tpot_for_config,
    get_thpt_for_config,
)
from evaluation.plots.utils.plot_utils import (
    SYS_NAME,
    crop_margins,
)

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10


def plot_ttft(
    qps_list: list[float],
    offload_ttfts: dict[float, list[float]],
    disagg_ttfts: list[float],
    output_path: str,
):
    # fig, ax = pyplot.subplots(figsize=(3.2, 2)) # 1/3 column
    fig, ax = pyplot.subplots(figsize=PlotConfig.figsize) # 1/4 column

    disagg_ttfts_s = [t/1000 for t in disagg_ttfts]
    offload_ratio, offload_ttft = list(offload_ttfts.items())[0]
    offload_ttft_s = [t/1000 for t in offload_ttft]

    ax.plot(qps_list, disagg_ttfts_s, marker=PlotConfig.marker_style[0],
            color=PlotConfig.colors[0], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label="vLLM")
    ax.plot(qps_list, offload_ttft_s, marker=PlotConfig.marker_style[1],
            color=PlotConfig.colors[1], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label=SYS_NAME)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(5))

    pyplot.grid(axis='y', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)
    pyplot.grid(axis='x', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)

    pyplot.ylim([0, 12])

    # for idx, (offload_ratio, offload_ttft) in enumerate(offload_ttfts.items()):
    #     pyplot.plot(
    #         qps_list, offload_ttft, color=PlotConfig.colors[idx+1], label=f"offload (ratio={offload_ratio})"
    #     )
    pyplot.xlabel("Req rate (req/s)")
    pyplot.ylabel("mean TTFT (s)")
    pyplot.tight_layout()
    fig.legend(loc=(0.45, 0.3), frameon=False, markerfirst=False)
    pyplot.savefig(output_path)


def plot_tpot(
    qps_list: list[float],
    offload_tpots: dict[float, list[float]],
    disagg_tpots: list[float],
    output_path: str,
):
    print(f'{disagg_tpots=}')
    print(f'{offload_tpots=}')
    # fig, ax = pyplot.subplots(figsize=(3.2, 2)) # 1/3 column
    fig, ax = pyplot.subplots(figsize=PlotConfig.figsize) # 1/4 column

    offload_ratio, offload_tpot = list(offload_tpots.items())[0]

    ax.plot(qps_list, disagg_tpots, marker=PlotConfig.marker_style[0],
            color=PlotConfig.colors[0], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label="vLLM")
    ax.plot(qps_list, offload_tpot, marker=PlotConfig.marker_style[1],
            color=PlotConfig.colors[1], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label=SYS_NAME)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    pyplot.grid(axis='y', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)
    pyplot.grid(axis='x', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)

    pyplot.ylim([20, 90])

    ax.set_xlabel("Req rate (req/s)")
    ax.set_ylabel("Mean TPOT (ms)")
    pyplot.tight_layout()
    fig.legend(loc=(0.45, 0.3), frameon=False, markerfirst=False)
    pyplot.savefig(output_path)


def plot_stable_thpt(
    qps_list: list[float],
    offload_thpts: dict[float, list[float]],
    disagg_thpts: list[float],
    output_path: str,
):
    # fig, ax = pyplot.subplots(figsize=(3.2, 2)) # 1/3 column
    fig, ax = pyplot.subplots(figsize=PlotConfig.figsize) # 1/4 column

    offload_ratio, offload_thpt = list(offload_thpts.items())[0]

    offload_thpt_per_s = [x * 1000 for x in offload_thpt]
    disagg_thpts_per_s = [x * 1000 for x in disagg_thpts]

    print(f'{offload_thpt_per_s=}')
    print(f'{disagg_thpts_per_s=}')
    speedup = [
        offload_thpt_per_s[i]/disagg_thpts_per_s[i]
        for i in range(len(disagg_thpts_per_s))
    ]
    print(f'{speedup=}, {max(speedup)=}, {min(speedup)=}')

    ax.plot(qps_list, disagg_thpts_per_s, marker=PlotConfig.marker_style[0],
            color=PlotConfig.colors[0], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label="vLLM")
    ax.plot(qps_list, offload_thpt_per_s, marker=PlotConfig.marker_style[1],
            color=PlotConfig.colors[1], markersize=PlotConfig.markersize,
            markerfacecolor='none', markeredgewidth=PlotConfig.markeredgewidth,
            label=SYS_NAME)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(400))

    pyplot.grid(axis='y', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)
    pyplot.grid(axis='x', color='lightgray', linestyle='--', linewidth=PlotConfig.grid_linewidth)

    pyplot.ylim([400, 1000])

    ax.set_xlabel("Req rate (req/s)")
    ax.set_ylabel("Throughput (tokens/s)")
    pyplot.tight_layout()
    # fig.legend(loc=(0.52, 0.3), frameon=False, markerfirst=False)
    fig.legend(loc=(0.32, 0.67), frameon=False)
    pyplot.savefig(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mooncake")
    parser.add_argument(
        "--qps",
        nargs="+",
        type=str,
        default=["1", "1.5", "2", "2.5", "3", "3.5", "4"],
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--numreqs", type=int, default=60)
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/plots/dataset_performance"
    )

    parser.add_argument("--disagg-dir", type=str, default="outputs/e2e_perf/disagg/3p1d")
    parser.add_argument("--offload-dir", type=str, default="outputs/e2e_perf/offload/3p1d")
    parser.add_argument("--num-execs", type=int, default=3)
    parser.add_argument("--crop-margin", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    configs = [f"{args.dataset}-qps_{qps}-numreqs_{args.numreqs}" for qps in args.qps]
    model_name = Path(args.model).name
    qps_float_list = [float(qps) for qps in args.qps]
    offload_ratios: list[float] = [""]
    offload_dirs = {
        ratio: f"{args.offload_dir}/{args.model}" for ratio in offload_ratios
    }
    offload_ttfts = {
        ratio: [
            get_ttft_for_config(config, offload_dirs[ratio], args.num_execs)
            for config in configs
        ]
        for ratio in offload_ratios
    }
    disagg_dir = f"{args.disagg_dir}/{args.model}"
    disagg_ttfts = [
        get_ttft_for_config(config, disagg_dir, args.num_execs)
        for config in configs
    ]
    print(offload_ttfts, disagg_ttfts)
    plot_ttft(qps_float_list, offload_ttfts, disagg_ttfts, f"{args.output_dir}/{args.dataset}_{model_name}_ttft.pdf")

    offload_tpots = {
        ratio: [
            get_tpot_for_config(config, offload_dirs[ratio], args.num_execs)
            for config in configs
        ]
        for ratio in offload_ratios
    }
    disagg_tpots = [
        get_tpot_for_config(config, disagg_dir, args.num_execs)
        for config in configs
    ]
    print(offload_tpots, disagg_tpots)
    plot_tpot(qps_float_list, offload_tpots, disagg_tpots, f"{args.output_dir}/{args.dataset}_{model_name}_tpot.pdf")

    offload_thpts = {
        ratio: [
            get_thpt_for_config(config, offload_dirs[ratio], args.num_execs)
            for config in configs
        ]
        for ratio in offload_ratios
    }
    disagg_thpts = [
        get_thpt_for_config(config, disagg_dir, args.num_execs)
        for config in configs
    ]
    print(offload_thpts, disagg_thpts)
    plot_stable_thpt(
        qps_float_list, offload_thpts, disagg_thpts, f"{args.output_dir}/{args.dataset}_{model_name}_thpt.pdf"
    )
    if args.crop_margin:
        crop_margins(f"{args.output_dir}/{args.dataset}_{model_name}_ttft.pdf")
        crop_margins(f"{args.output_dir}/{args.dataset}_{model_name}_tpot.pdf")
        crop_margins(f"{args.output_dir}/{args.dataset}_{model_name}_thpt.pdf")
