import dataclasses
import marshal


@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """

    prompt: str
    prompt_len: int
    output_len: int


@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """

    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: list[TestRequest]

    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))

    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )
