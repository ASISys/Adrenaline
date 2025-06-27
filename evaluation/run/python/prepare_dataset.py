import argparse
import glob
import json
import numpy
import os
import pandas
import random
import sys
import torch
import tqdm

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import Optional
from benchmarks.dataset_utils import TestRequest, Dataset


def prepare_sharegpt(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
):
    # Load the dataset.
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    requests: list[TestRequest] = []
    for data in tqdm.tqdm(dataset[:1000]):
        num_conversations = len(data["conversations"])

        # Filter out the conversations with less than args.sharegpt_min_turns turns.
        if num_conversations < args.sharegpt_min_turns or \
            num_conversations < args.sharegpt_min_prompt_turns + 1:
            continue

        num_prompt_turns = random.randint(
            args.sharegpt_min_prompt_turns,
            min(num_conversations - 1, args.sharegpt_max_prompt_turns)
        )

        prompt = "\n".join([data["conversations"][i]["value"] for i in range(num_prompt_turns)])
        completion = data["conversations"][num_prompt_turns]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids

        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)

        if prompt_len < args.min_model_len or output_len < args.min_model_len:
            continue
        if prompt_len + output_len > args.max_model_len:
            continue

        requests.append(TestRequest(prompt, prompt_len, output_len))

    if len(requests) < 500:
        requests += requests
    return Dataset(f"sharegpt", requests)


def prepare_mooncake(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_model_len: Optional[int] = None,
):
    with open(dataset_path) as f:
        lines = f.readlines()
    data_records = [json.loads(line) for line in lines[:1000]]
    input_lengths = [data["input_length"] for data in data_records]
    output_lengths = [data["output_length"] for data in data_records]
    if max_model_len is not None:
        input_lengths, output_lengths = zip(*[
            (input_length, output_length)
            for input_length, output_length in zip(input_lengths, output_lengths)
            if input_length + output_length < max_model_len
        ])
    offsets = [
        numpy.random.randint(0, tokenizer.vocab_size) for _ in range(len(data_records))
    ]
    prompts = [
        tokenizer.decode([(offset + i + j) % tokenizer.vocab_size for j in range(input_length)])
        for i, (offset, input_length) in enumerate(zip(offsets, input_lengths))
    ]
    requests = [
        TestRequest(prompt, input_length, output_length)
        for prompt, input_length, output_length in zip(
            prompts, input_lengths, output_lengths
        )
    ]
    return Dataset("mooncake", requests)


def prepare_open_thoughts():
    requests: list[TestRequest] = []
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train")
    dataset = ds[:1000]
    for sys_prompt, conversations in tqdm.tqdm(
        zip(dataset["system"], dataset["conversations"])
    ):
        num_conversations = len(conversations)
        if num_conversations < 2:
            continue
        turn_index = random.randint(0, num_conversations // 2 - 1)
        prompt_idx = turn_index * 2
        completion_idx = prompt_idx + 1
        prompt = sys_prompt + conversations[prompt_idx]["value"]
        completion = conversations[completion_idx]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        role = conversations[completion_idx]["from"]

        if prompt_len + output_len > args.max_model_len:
            continue
        if role != "assistant":
            print(f"{turn_index=}, {prompt_idx=}, {completion_idx=}")
            continue
        requests.append(TestRequest(prompt, prompt_len, output_len))

    return Dataset("open-thoughts", requests)


def prepare_random_dataset(
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
    input_len: int,
    output_len: int,
    range_ratio: float,
    prefix_len: int = 0,
    output_ds_name: str = "random",
):
    prefix_token_ids = numpy.random.randint(
        0, tokenizer.vocab_size, size=prefix_len
    ).tolist()

    input_lens = numpy.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = numpy.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = numpy.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])
        requests.append(
            TestRequest(prompt, prefix_len + int(input_lens[i]), int(output_lens[i]))
        )

    return Dataset(output_ds_name, requests)


def read_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    name: str,
    args: argparse.Namespace,
) -> Dataset:
    """
    read_dataset: Read the given dataset and return a list of TestRequest.
    """
    if name.lower() == "sharegpt":
        dataset = prepare_sharegpt(dataset_path, tokenizer, args)
        return dataset

    elif name.lower() == "open-thoughts":
        dataset = prepare_open_thoughts()
        return dataset

    elif name.lower() == "mooncake":
        dataset = prepare_mooncake(dataset_path, tokenizer, args.max_model_len)
        return dataset

    elif name.lower() == "random":
        dataset = prepare_random_dataset(
            args.random_num_prompts,
            tokenizer,
            args.random_input_len,
            args.random_output_len,
            args.random_range_ratio,
            output_ds_name=f"random_p{args.random_input_len}_d{args.random_output_len}",
        )
        return dataset

    elif name.lower() == "alpaca":
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # extract the input and output
        dataset = [
            (data["instruction"] + data["input"], data["output"]) for data in dataset
        ]

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: list[TestRequest] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))

        return Dataset("alpaca", filtered_dataset)

    elif name.lower() == "mmlu":
        dataset = []
        choices = ["A", "B", "C", "D"]
        data_path = dataset_path
        subjects = sorted(
            [
                f.split("_test.csv")[0]
                for f in os.listdir(os.path.join(data_path, "test"))
                if "_test.csv" in f
            ]
        )

        for sub in subjects:
            test_df = pandas.read_csv(
                os.path.join(data_path, "test", sub + "_test.csv"), header=None
            )
            for i in range(test_df.shape[0]):
                prompt = test_df.iloc[i, 0]
                k = test_df.shape[1] - 2
                for j in range(k):
                    prompt += "\n{}. {}".format(choices[j], test_df.iloc[i, j + 1])
                prompt += "\nAnswer:"
                output = test_df.iloc[i, k + 1]
                dataset.append((prompt, output))

        print("MMLU dataset size:", len(dataset))

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: list[TestRequest] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))

        return Dataset("mmlu", filtered_dataset)

    elif name.lower() == "longbench":
        # find all .jsonl files under the dataset_path
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))

        filtered_dataset = []
        for file in tqdm.tqdm(files):
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    data = json.loads(line)

                    context = data["context"][:40000]    # truncate to the first 40000 chars to reduce tokenization time
                    context_token_ids = tokenizer(context).input_ids
                    answer_token_ids = tokenizer(data["answers"][0]).input_ids
                    context_len = len(context_token_ids)
                    answer_len = len(answer_token_ids)
                    if context_len + answer_len > args.max_model_len:
                        continue
                    if context_len < args.min_model_len or answer_len < args.min_model_len:
                        continue

                    context_len_allowed = min(2040 - answer_len, random.randint(args.longbench_min_prompt_len, args.longbench_max_prompt_len))
                    context_token_ids = context_token_ids[:context_len_allowed]

                    filtered_dataset.append(TestRequest(
                        tokenizer.decode(context_token_ids),
                        len(context_token_ids),
                        answer_len
                    ))

        # return Dataset(f"longbench-mipl-{args.longbench_min_prompt_len}-mxpl-{args.longbench_max_prompt_len}", filtered_dataset)
        return Dataset(f"longbench", filtered_dataset)

    elif name.lower() == "humaneval":
        filtered_dataset = []
        with open(dataset_path, "r") as f:
            for line in f.readlines():
                if line.strip() == "": continue
                data = json.loads(line)

                context = data["prompt"]
                context_token_ids = tokenizer(context).input_ids
                answer = data["canonical_solution"]
                answer_token_ids = tokenizer(answer).input_ids

                context_len = len(context_token_ids)
                answer_len = len(answer_token_ids)
                if context_len + answer_len > args.max_model_len:
                    continue
                if context_len < args.min_model_len or answer_len < args.min_model_len:
                    continue

                filtered_dataset.append(TestRequest(
                    context,
                    len(context_token_ids),
                    len(answer_token_ids)
                ))

        # Copy the dataset for 10 times since it's too small.
        filtered_dataset = filtered_dataset * 10

        return Dataset("humaneval", filtered_dataset)

    elif name.lower() == "sharegpt52k":
        files = glob.glob(os.path.join(dataset_path, "*.json"))
        result: list[TestRequest] = []
        for file in files:
            with open(file, "r") as f:
                dataset = json.load(f)
            for data in tqdm.tqdm(dataset[:10000]):
                num_conversations = len(data["conversations"])
                if num_conversations < 2:
                    continue
                turn_index = random.randint(0, num_conversations // 2 - 1)
                prompt_idx = turn_index * 2
                completion_idx = prompt_idx + 1
                prompt = data["conversations"][prompt_idx]["value"]
                completion = data["conversations"][completion_idx]["value"]
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(completion).input_ids
                prompt_len = len(prompt_token_ids)
                output_len = len(completion_token_ids)
                role = data["conversations"][completion_idx]["from"]

                if prompt_len + output_len > args.max_model_len:
                    continue

                if prompt_len < args.min_model_len or output_len < args.min_model_len:
                    continue

                if role != "gpt":
                    print(f"{turn_index=}, {prompt_idx=}, {completion_idx=}, {data["id"]=}")
                    continue

                result.append(TestRequest(prompt, prompt_len, output_len))

        return Dataset(f"sharegpt52k", result)

    elif name.lower() == "sharegpt4":
        ds = load_dataset("shibing624/sharegpt_gpt4")
        dataset = ds["train"]
        result: list[TestRequest] = []
        for data in tqdm.tqdm(dataset[:10000]["conversations"]):
            num_conversations = len(data)
            if num_conversations < 2:
                continue
            turn_index = random.randint(0, num_conversations // 2 - 1)
            prompt_idx = turn_index * 2
            completion_idx = prompt_idx + 1
            prompt = data[prompt_idx]["value"]
            completion = data[completion_idx]["value"]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids)
            role = data[completion_idx]["from"]

            print(f"{prompt_len=}, {output_len=}, {role=}")
            if prompt_len + output_len > args.max_model_len:
                continue
            if prompt_len < args.min_model_len or output_len < args.min_model_len:
                continue

            if role != "gpt":
                print(f"{turn_index=}, {prompt_idx=}, {completion_idx=}")
                continue

            result.append(TestRequest(prompt, prompt_len, output_len))

        return Dataset(f"sharegpt4", result)

    elif name.lower() == "leval":
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))

        num_lines = sum([
            sum([
                len(json.loads(line)["instructions"])
                for line in open(filename, "r").readlines()
            ])
            for filename in files
        ])

        dataset = []
        pbar = tqdm.tqdm(total=num_lines)
        for file in files:
            print(f"Processing {file}")
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    data = json.loads(line)

                    input = data["input"]
                    input_len = len(tokenizer.tokenize(input))
                    for (instruction, output) in zip(data["instructions"], data["outputs"]):
                        prompt_len = input_len + len(tokenizer.tokenize(instruction))
                        output_len = len(tokenizer.tokenize(output))
                        print(f"{prompt_len=}, {output_len=}")
                        dataset.append(TestRequest(
                            input+instruction,
                            prompt_len,
                            output_len
                        ))
                        pbar.update(1)

        pbar.close()
        return Dataset(name, dataset)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sharegpt")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", default=32768, type=int)
    parser.add_argument("--min-model-len", default=4, type=int)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--sharegpt-min-turns", type=int, default=3)
    parser.add_argument("--sharegpt-min-prompt-turns", type=int, default=1)
    parser.add_argument("--sharegpt-max-prompt-turns", type=int, default=1000)

    parser.add_argument("--longbench-min-prompt-len", type=int, default=1900)
    parser.add_argument("--longbench-max-prompt-len", type=int, default=2048)

    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--random-range-ratio", type=float, default=1)
    parser.add_argument("--random-num-prompts", type=int, default=1000)
    args = parser.parse_args()
    return args


def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    if not args.force and os.path.exists(args.output_path):
        print(f"Output file {args.output_path} already exists. Use --force to overwrite.")
        sys.exit(0)
    set_random_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    dataset = read_dataset(args.dataset_path, tokenizer, args.dataset, args)
    dataset_path = args.dataset if args.dataset_path is None else args.dataset_path
    print(f"Loaded {len(dataset.reqs)} TestRequests from dataset {dataset_path}")
    dataset.dump(args.output_path)
    print(f"Saved to {args.output_path}")
