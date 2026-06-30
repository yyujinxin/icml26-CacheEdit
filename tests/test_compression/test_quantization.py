import unittest

import torch

from cache_edit.compression.activation_compressor import (
    _quantization_name,
    _quantization_rows_per_frame,
)
from cache_edit.compression.pipeline.quantization import (
    CWQuantization,
    GWOutlierQuantization,
    GWQuantization,
)
from cache_edit.models.flux.cache_manager import FluxCacheManager


def _roundtrip(step, tensor):
    data = {"data": tensor.clone()}
    encoded = step.forward(data)
    encoded["shape"] = tensor.shape
    decoded = step.backward(encoded)
    return decoded["data"]


class QuantizationTest(unittest.TestCase):
    def test_groupwise_quantization_roundtrip_has_finite_error(self):
        torch.manual_seed(0)
        tensor = (torch.randn(7, 64, dtype=torch.float16) * 3.0).clamp(
            -6.0, 6.0
        )

        recovered = _roundtrip(GWQuantization(groupsize=16), tensor)

        self.assertEqual(recovered.shape, tensor.shape)
        self.assertTrue(torch.isfinite(recovered).all().item())
        self.assertLess(
            (recovered.float() - tensor.float()).abs().max().item(), 0.08
        )

    def test_larger_group_size_uses_fewer_scale_rows(self):
        tensor = torch.randn(5, 64, dtype=torch.float16)

        gw16_data = GWQuantization(groupsize=16).forward({"data": tensor.clone()})
        gw64_data = GWQuantization(groupsize=64).forward({"data": tensor.clone()})

        self.assertEqual(gw16_data["scale"].shape[0], 20)
        self.assertEqual(gw64_data["scale"].shape[0], 5)

    def test_groupwise_outlier_quantization_reduces_error(self):
        torch.manual_seed(4)
        tensor = torch.randn(16, 64, dtype=torch.float16) * 3.0
        tensor[0, 0] = 80.0
        tensor[3, 17] = -70.0

        base = _roundtrip(GWQuantization(groupsize=64), tensor)
        outlier = _roundtrip(
            GWOutlierQuantization(groupsize=64, outlier_ratio=0.01),
            tensor,
        )

        base_err = (base.float() - tensor.float()).abs()
        outlier_err = (outlier.float() - tensor.float()).abs()
        self.assertLess(outlier_err.max().item(), base_err.max().item())
        self.assertLess(outlier_err.pow(2).mean().item(), base_err.pow(2).mean().item())

    def test_groupwise_quantization_stores_rounded_zero_point(self):
        tensor = torch.tensor(
            [[-2.0, -1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 9.0]],
            dtype=torch.float16,
        )

        encoded = GWQuantization(groupsize=4).forward({"data": tensor.clone()})

        self.assertTrue(
            torch.equal(
                encoded["offset"].flatten().cpu(),
                torch.tensor([102.0, -204.0]),
            )
        )

    def test_channelwise_quantization_roundtrip_has_finite_error(self):
        torch.manual_seed(1)
        tensor = torch.randn(9, 64, dtype=torch.float16)

        recovered = _roundtrip(CWQuantization(), tensor)

        self.assertEqual(recovered.shape, tensor.shape)
        self.assertTrue(torch.isfinite(recovered).all().item())
        self.assertLess(
            (recovered.float() - tensor.float()).abs().max().item(), 0.04
        )

    def test_explicit_none_quant_group_size_means_channelwise(self):
        self.assertEqual(_quantization_name("lossless", 256), "gw256")
        self.assertEqual(_quantization_name("lossless", 64, None), "cw")
        self.assertEqual(_quantization_name("lossless", 64, 64), "gw64")
        self.assertEqual(_quantization_name("lossless", 64, 64, 0.001), "gwo64")
        self.assertEqual(_quantization_name("hevc", 64, 64), "gw64")

    def test_rows_per_frame_parses_group_size_from_quantization_name(self):
        self.assertEqual(_quantization_rows_per_frame("gw16", 5, 64), 20)
        self.assertEqual(_quantization_rows_per_frame("gwo16", 5, 64), 20)
        self.assertEqual(_quantization_rows_per_frame("gw64", 5, 64), 5)
        self.assertEqual(_quantization_rows_per_frame("cw", 5, 64), 5)

    def test_quant_error_probe_prefers_smaller_groups(self):
        torch.manual_seed(2)
        manager = FluxCacheManager(
            use_activation_cache=False,
            use_compression=False,
            compression_quant_error_probe_groups=[16, 64],
            compression_quant_error_probe_max_rows=0,
        )
        tensor = torch.randn(8, 64, dtype=torch.float16) * 4.0

        qg16 = manager._quant_error_probe_for_tensor(tensor, 16)
        qg64 = manager._quant_error_probe_for_tensor(tensor, 64)

        self.assertEqual(qg16["status"], "ok")
        self.assertEqual(qg64["status"], "ok")
        self.assertLessEqual(qg16["mse_sum"], qg64["mse_sum"])
        self.assertGreater(qg16["metadata_bytes"], qg64["metadata_bytes"])

    def test_quant_error_probe_outlier_ratio_reduces_error_with_metadata(self):
        torch.manual_seed(5)
        manager = FluxCacheManager(
            use_activation_cache=False,
            use_compression=False,
            compression_quant_error_probe_groups=[64],
            compression_quant_error_probe_outlier_ratios=[0, 0.01],
            compression_quant_error_probe_max_rows=0,
        )
        tensor = torch.randn(16, 64, dtype=torch.float16) * 3.0
        tensor[0, 0] = 90.0
        tensor[4, 12] = -85.0

        base = manager._quant_error_probe_for_tensor(tensor, 64, 0.0)
        outlier = manager._quant_error_probe_for_tensor(tensor, 64, 0.01)

        self.assertEqual(base["status"], "ok")
        self.assertEqual(outlier["status"], "ok")
        self.assertLess(outlier["mse_sum"], base["mse_sum"])
        self.assertGreater(outlier["metadata_bytes"], base["metadata_bytes"])

    def test_quant_error_probe_appears_in_report(self):
        torch.manual_seed(3)
        manager = FluxCacheManager(
            use_activation_cache=False,
            use_compression=False,
            compression_quant_error_probe_groups=[16, 64, 0],
            compression_quant_error_probe_outlier_ratios=[0, 0.01],
            compression_quant_error_probe_max_rows=4,
        )
        manager.current_round = 1
        manager.current_step = 5
        manager._record_quant_error_probe_group(
            stream="single",
            step=5,
            layer_indices=[0, 1],
            tensors=[
                torch.randn(8, 64, dtype=torch.float16),
                torch.randn(8, 64, dtype=torch.float16),
            ],
        )

        report = manager.get_compression_report(include_records=True)
        summary = report["summary"]

        self.assertTrue(summary["quant_error_probe_enabled"])
        self.assertIn("qg16", summary["quant_error_probe_by_quantization"])
        self.assertIn("qg16_o0p01", summary["quant_error_probe_by_quantization"])
        self.assertIn("qg64", summary["quant_error_probe_by_quantization"])
        self.assertIn("cw", summary["quant_error_probe_by_quantization"])
        self.assertEqual(len(report["quant_error_probe_records"]), 5)


if __name__ == "__main__":
    unittest.main()
