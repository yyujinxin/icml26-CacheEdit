from .definitions import Step
from typing import Dict, Any, List
import torch

class Transpose(Step):
    def __init__(self):
        super(Transpose, self).__init__("Transpose",
                                             required_keys=["data"],
                                             yield_keys=["data"],
                                             preserve_keys=[],
                                             )

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict["data"] = data_dict["data"].T
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict["data"] = data_dict["data"].T
        return data_dict

class CWQuantization(Step):
    def __init__(self, unsafe_scale: bool = False):
        super(CWQuantization, self).__init__("CWQuantization",
                                             required_keys=["data", 'shape'],
                                             yield_keys=["data", 'scale', 'offset'],
                                             preserve_keys=[],
                                             )
        self.unsafe_scale = unsafe_scale

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        orig_shape = data_dict["data"].shape
        tensor = data_dict["data"].float()
        tensor = tensor.view(-1, orig_shape[-1])
        min_val, _ = tensor.min(dim=1)
        max_val, _ = tensor.max(dim=1)
        if self.unsafe_scale:
            scale = (max_val - min_val) / 255
        else:
            scale = (max_val - min_val).clamp(min=1e-5) / 255
        offset = min_val
        scale = scale.unsqueeze(1)
        offset = offset.unsqueeze(1)
        tensor_q = torch.clamp(torch.round((tensor - offset) / scale), 0, 255).to(torch.uint8)

        # reshape back
        tensor_q = tensor_q.view(orig_shape)
        data_dict["data"] = tensor_q
        data_dict["scale"] = scale
        data_dict["offset"] = offset
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        tensor_q = data_dict["data"]
        scale = data_dict["scale"]
        offset = data_dict["offset"]
        tensor_q = tensor_q.float()
        tensor_q = tensor_q.view(-1, tensor_q.shape[-1])
        tensor = tensor_q * scale + offset
        tensor = tensor.view(data_dict["shape"])
        data_dict["data"] = tensor
        return data_dict


class GWQuantization(Step):
    def __init__(self, groupsize=128, unsafe_scale: bool = False):
        super(GWQuantization, self).__init__("GWQuantization",
                                             required_keys=["data", 'shape'],
                                             yield_keys=["data", 'scale', 'offset'],
                                             preserve_keys=[],
                                             )
        self.unsafe_scale = unsafe_scale
        self.groupsize = groupsize

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        orig_shape = data_dict["data"].shape
        tensor = data_dict["data"].float()
        tensor = tensor.view(-1, self.groupsize)
        min_val, _ = tensor.min(dim=1)
        max_val, _ = tensor.max(dim=1)
        if self.unsafe_scale:
            scale = (max_val - min_val) / 255
        else:
            scale = (max_val - min_val).clamp(min=1e-5) / 255
        zero = torch.round(-min_val / scale)
        scale = scale.unsqueeze(1)
        zero = zero.unsqueeze(1)
        tensor_q = torch.clamp(torch.round(tensor / scale) + zero, 0, 255).to(torch.uint8)

        # reshape back
        tensor_q = tensor_q.view(orig_shape)
        data_dict["data"] = tensor_q
        data_dict["scale"] = scale
        data_dict["offset"] = zero
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        tensor_q = data_dict["data"]
        scale = data_dict["scale"]
        offset = data_dict["offset"]
        tensor_q = tensor_q.float()
        tensor_q = tensor_q.view(-1, self.groupsize)
        # tensor_q = tensor_q.view(-1, tensor_q.shape[-1])
        tensor = scale * (tensor_q - offset)
        tensor = tensor.view(data_dict["shape"])
        data_dict["data"] = tensor
        return data_dict


class GWOutlierQuantization(Step):
    def __init__(
        self,
        groupsize=128,
        outlier_ratio: float = 0.0,
        unsafe_scale: bool = False,
    ):
        super(GWOutlierQuantization, self).__init__(
            "GWOutlierQuantization",
            required_keys=["data", 'shape'],
            yield_keys=["data", 'scale', 'offset', 'outlier_indices', 'outlier_residuals'],
            preserve_keys=[],
        )
        self.unsafe_scale = unsafe_scale
        self.groupsize = groupsize
        self.outlier_ratio = max(0.0, float(outlier_ratio))

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        orig_shape = data_dict["data"].shape
        tensor = data_dict["data"].float()
        grouped = tensor.reshape(-1, self.groupsize)
        min_val, _ = grouped.min(dim=1)
        max_val, _ = grouped.max(dim=1)
        if self.unsafe_scale:
            scale = (max_val - min_val) / 255
        else:
            scale = (max_val - min_val).clamp(min=1e-5) / 255
        zero = torch.round(-min_val / scale)
        scale_2d = scale.unsqueeze(1)
        zero_2d = zero.unsqueeze(1)
        tensor_q = torch.clamp(
            torch.round(grouped / scale_2d) + zero_2d,
            0,
            255,
        ).to(torch.uint8)

        outlier_indices = torch.empty(0, dtype=torch.int32, device=tensor.device)
        outlier_residuals = torch.empty(0, dtype=torch.float16, device=tensor.device)
        if self.outlier_ratio > 0.0 and grouped.numel() > 0:
            restored = scale_2d * (tensor_q.float() - zero_2d)
            residual = grouped - restored
            k = int(round(float(grouped.numel()) * self.outlier_ratio))
            k = max(0, min(k, int(grouped.numel())))
            if k > 0:
                _, indices = torch.topk(residual.abs().reshape(-1), k=k, largest=True)
                indices = indices.to(torch.int32)
                outlier_indices = indices
                outlier_residuals = residual.reshape(-1).index_select(
                    0,
                    indices.to(torch.long),
                ).to(torch.float16)

        data_dict["data"] = tensor_q.reshape(orig_shape)
        data_dict["scale"] = scale_2d
        data_dict["offset"] = zero_2d
        data_dict["outlier_indices"] = outlier_indices
        data_dict["outlier_residuals"] = outlier_residuals
        data_dict["outlier_ratio"] = float(self.outlier_ratio)
        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        tensor_q = data_dict["data"]
        scale = data_dict["scale"]
        offset = data_dict["offset"]
        tensor_q = tensor_q.float().reshape(-1, self.groupsize)
        tensor = scale * (tensor_q - offset)

        outlier_indices = data_dict.get("outlier_indices")
        outlier_residuals = data_dict.get("outlier_residuals")
        if (
            isinstance(outlier_indices, torch.Tensor)
            and isinstance(outlier_residuals, torch.Tensor)
            and outlier_indices.numel() > 0
            and outlier_residuals.numel() > 0
        ):
            flat = tensor.reshape(-1)
            flat[outlier_indices.to(device=flat.device, dtype=torch.long)] += (
                outlier_residuals.to(device=flat.device, dtype=flat.dtype)
            )

        tensor = tensor.reshape(data_dict["shape"])
        data_dict["data"] = tensor
        return data_dict


class CWOQuantization(Step):
    def __init__(self, outlier_vars_to_center: float = 3, clip_vars_to_center: float = 2):
        super(CWOQuantization, self).__init__("CWOQuantization",
                                              required_keys=["data", 'shape'],
                                              yield_keys=["data", 'scale', 'offset'],
                                              preserve_keys=[],
                                              )
        self.outlier_vars_to_center = outlier_vars_to_center
        self.clip_vars_to_center = clip_vars_to_center

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        tensor = data_dict["data"]
        orig_shape = data_dict["data"].shape
        tensor = tensor.view(-1, orig_shape[-1])
        means = tensor.mean(dim=0)
        vars = torch.sqrt(tensor.var(dim=0))
        # estimate the outlier threshold using the assumption of a normal distribution
        outlier_threshold_up = means + self.outlier_vars_to_center * vars
        outlier_threshold_down = means - self.outlier_vars_to_center * vars
        clip_threshold_up = means + self.clip_vars_to_center * vars
        clip_threshold_down = means - self.clip_vars_to_center * vars

        # extract outliers and put them into a sparse tensor
        outlier_mask_up = tensor > outlier_threshold_up.unsqueeze(0)
        outlier_mask_down = tensor < outlier_threshold_down.unsqueeze(0)
        outlier_mask = outlier_mask_up | outlier_mask_down

        # Extract outliers
        outliers = tensor[outlier_mask]
        # Construct sparse tensor
        indices = outlier_mask.nonzero().t()
        values = outliers
        size = tensor.size()
        sparse_outliers = torch.sparse_coo_tensor(indices, values, size)
        # actual sparse ratio
        sparse_ratio = sparse_outliers._nnz() / tensor.numel()
        # force the outliers in tensor to 0
        tensor[outlier_mask] = 0

        # clip
        tensor = tensor.clamp(min=clip_threshold_down.unsqueeze(0), max=clip_threshold_up.unsqueeze(0))

        # Channel-wise quantization
        min_val, _ = tensor.min(dim=0)
        max_val, _ = tensor.max(dim=0)
        scale = max_val - min_val

        tensor_scaled = (tensor - min_val) / scale
        tensor_q = torch.round(tensor_scaled * 255).to(torch.uint8)
        # reshape back
        tensor_q = tensor_q.view(orig_shape)
        data_dict["data"] = tensor_q
        data_dict["scale"] = scale
        data_dict["offset"] = min_val
        data_dict["sparse_ratio"] = sparse_ratio
        data_dict["outliers"] = sparse_outliers

        return data_dict

    def backward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        tensor_q = data_dict["data"]
        orig_shape = tensor_q.shape
        scale = data_dict["scale"]
        offset = data_dict["offset"]
        outliers = data_dict["outliers"]

        # Convert tensor back from quantization
        tensor_unq = tensor_q.float() / 255
        tensor_unscaled = tensor_unq * scale + offset

        # Reshape to original shape if needed (useful if the tensor was reshaped during forward)
        tensor_unscaled = tensor_unscaled.view(orig_shape)

        # Convert sparse tensor to dense format and reshape
        outliers_dense = outliers.to_dense().view(orig_shape)
        tensor_unscaled += outliers_dense

        # Update the data dictionary
        data_dict["data"] = tensor_unscaled

        return data_dict
