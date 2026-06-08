from typing import Dict, Any, List
from .definitions import Step
import torch
import matplotlib
import matplotlib.pyplot as plt
import os


class CheckShape(Step):
    def __init__(self, expected_shape: List[int]):
        super(CheckShape, self).__init__("CheckShape", required_keys=["data"], yield_keys=['data', 'shape'])
        self.expected_shape = list(expected_shape)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert list(data_dict["data"].shape) == self.expected_shape, "Expect shape {}, got shape {}".format(
            self.expected_shape, data_dict["data"].shape)
        data_dict["shape"] = data_dict["data"].shape
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert list(data_dict["data"].shape) == self.expected_shape, "Expect shape {}, got shape {}".format(
            self.expected_shape, data_dict["data"].shape)
        return data_dict

    def __repr__(self):
        return "CheckShape({})".format(self.expected_shape)


class DumpAsImage(Step):
    def __init__(self, dump_recovered=False, base_dir="figs/dumped_img"):
        super(DumpAsImage, self).__init__("DumpAsImage", required_keys=["data", "name"], yield_keys=["data"])
        self.base_dir = base_dir
        self.dump_recovered = dump_recovered

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        print(data_dict.keys())
        name = data_dict["name"] if "name" in data_dict else "dumped"
        num_tiles = x.shape[0]
        for i in range(num_tiles):
            this_tile = torch.clone(x[i].contiguous()).detach().cpu()
            this_tile = this_tile.permute(0, 2, 3, 1)[:, :, :, :1].repeat(1, 1, 1, 3)
            this_tile = this_tile.numpy()
            # iterate through the first dim
            for j in range(this_tile.shape[0]):
                out_path = os.path.join(self.base_dir, f"{name}_t{i}_f{j}.png")
                matplotlib.image.imsave(out_path, this_tile[j])
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.dump_recovered:
            return data_dict
        x = data_dict["data"]
        print(data_dict.keys())
        name = data_dict["name"] if "name" in data_dict else "dumped"
        num_tiles = x.shape[0]
        for i in range(num_tiles):
            this_tile = torch.clone(x[i].contiguous()).detach().cpu()
            this_tile = this_tile.permute(0, 2, 3, 1)[:, :, :, :1].repeat(1, 1, 1, 3)
            this_tile = this_tile.numpy()
            # iterate through the first dim
            for j in range(this_tile.shape[0]):
                out_path = os.path.join(self.base_dir, f"{name}_t{i}_f{j}_recovered.png")
                matplotlib.image.imsave(out_path, this_tile[j])
        return data_dict



class ScatterPlot(Step):
    def __init__(self, base_dir="figs/scatter", points_count=100000):
        super(ScatterPlot, self).__init__("ScatterPlot", required_keys=["data", "name"], yield_keys=["data"])
        self.base_dir = base_dir
        self.original_x = None
        self.indices = None
        self.points_count = points_count

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]
        numel = x.numel()
        self.indices = torch.randint(0, numel, (self.points_count,))
        self.original_x = x.detach().cpu().flatten().numpy()[self.indices]
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict["data"]

        recovered = x.detach().cpu().flatten().numpy()[self.indices]
        plt.clf()
        plt.figure(figsize=(10, 10), dpi=200)
        plt.scatter(self.original_x, recovered, s=2, alpha=0.2)
        plt.plot([min(self.original_x), max(self.original_x)], [min(self.original_x), max(self.original_x)], color='red')
        plt.xlabel("Original")
        plt.ylabel("Recovered")
        plt.title(data_dict["name"])
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f"{data_dict['name']}.png"))
        plt.close()

        self.indices = None
        self.original_x = None
        return data_dict
