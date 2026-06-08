from typing import Dict, Any, List
import torch

class Step:
    def __init__(self, name: str, required_keys: List[str], yield_keys: List[str], preserve_keys: List[str] = []):
        self.name = name
        self.required_keys = required_keys
        self.generate_keys = yield_keys
        self.preserve_keys = preserve_keys

    # abstract function for forward and backward
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def __repr__(self):
        return self.name


class Pipeline:
    def __init__(self, steps: List[Step]):
        self.steps = steps
        self.stats = {}

    def forward(self, x: torch.Tensor, name: str = None) -> Dict[str, Any]:
        data_dict = {"data": x}
        if name is not None:
            data_dict["name"] = name
        for step in self.steps:
            # check preserve keys
            data_dict = step.forward(data_dict)
            for k in step.preserve_keys:
                self.stats[k] = data_dict[k]
        return data_dict

    def forward_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps:
            data_dict = step.forward(data_dict)
            for k in step.preserve_keys:
                self.stats[k] = data_dict[k]
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        # reverse the steps
        for step in self.steps[::-1]:
            data_dict = step.backward(data_dict)
        return data_dict["data"]

    def backward_to_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps[::-1]:
            data_dict = step.backward(data_dict)
        return data_dict

    def forward_backward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backward(self.forward(x))

    def __repr__(self):
        ret = "Pipeline(\n"
        for step in self.steps:
            ret += "\t" + step.__repr__() + " -> \n"
        ret += ")"
        return ret


class Batch(Step):
    def __init__(self, pipeline: Pipeline,
                required_keys=['data'],
                yield_keys=["data"],
                preserve_keys=[],
                 ):
        super(Batch, self).__init__("Batch", required_keys, yield_keys, preserve_keys)
        self.pipeline = pipeline

    def forward(self, data_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.pipeline.forward_dict(data_dict) for data_dict in data_dicts]

    def backward(self, data_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.pipeline.backward_to_dict(data_dict) for data_dict in data_dicts]

    def __repr__(self):
        repr = self.pipeline.__repr__()
        # append \t to each line
        repr = repr.replace("\n", "\n\t")
        return "Batch(" + repr + ")"
