import argparse
import numpy

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from adrenaline.utils import csv_utils


def parse_field(line: str, field: str):
    assert field in line
    start_idx = line.find(field)
    sub_line = line[start_idx:]
    end_idx = sub_line.find(",")
    if end_idx != -1:
        sub_line = sub_line[:end_idx]
    else:
        sub_line = sub_line.strip()
    return sub_line.split("=")[1]


def parse_num_reqs(line: str):
    return int(parse_field(line, "num_requests"))


def parse_estimate_decode_time(line: str):
    return float(parse_field(line, "estimate decode_time"))


def parse_recent_decode_time(line: str):
    recent_decode_time = parse_field(line, "recent decode_time")
    if recent_decode_time == "None":
        return None
    return float(recent_decode_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="proxy.out")
    parser.add_argument("--output-file", type=str, default="adrenaline/utils/misc_time.csv")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    input_file: str = args.input_file
    output_file: str = args.output_file
    with open(input_file) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if "decode_time" in l]
    num_reqs = [parse_num_reqs(l) for l in lines]
    estimate_decode_times = [parse_estimate_decode_time(l) for l in lines]
    recent_decode_times = [parse_recent_decode_time(l) for l in lines]

    num_reqs = numpy.array(num_reqs)
    estimate_decode_times = numpy.array(estimate_decode_times)
    recent_decode_times = numpy.array(recent_decode_times)
    mask = recent_decode_times != None
    num_reqs = num_reqs[mask]
    estimate_decode_times = estimate_decode_times[mask]
    recent_decode_times = recent_decode_times[mask]
    delta_decode_times = recent_decode_times - estimate_decode_times

    model = RANSACRegressor()
    model.fit(num_reqs.reshape(-1, 1), delta_decode_times.reshape(-1, 1))
    inliers = model.inlier_mask_
    model_ransac = LinearRegression()
    model_ransac.fit(
        num_reqs[inliers].reshape(-1, 1), delta_decode_times[inliers].reshape(-1, 1)
    )
    coef = model_ransac.coef_[0][0]
    intercept = model_ransac.intercept_[0]
    csv_utils.save_to_csv(
        items=[{"delta_coef": coef, "delta_intercept": intercept}],
        headers=["delta_coef", "delta_intercept"],
        output_path=output_file,
    )
