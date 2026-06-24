#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvcodec.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<InputFormat>(m, "InputFormat")
        .value("YUV444", InputFormat::INPUT_YUV444)
        .value("ARGB", InputFormat::INPUT_ARGB)
        .value("NV12", InputFormat::INPUT_NV12);

    py::enum_<CodecType>(m, "CodecType")
        .value("H264", CodecType::CODEC_H264)
        .value("HEVC", CodecType::CODEC_HEVC);

    py::enum_<RateControlMode>(m, "RateControlMode")
        .value("ConstQP", RateControlMode::RC_CONSTQP)
        .value("CBR", RateControlMode::RC_CBR)
        .value("VBR", RateControlMode::RC_VBR);

    py::enum_<PresetType>(m, "PresetType")
        .value("P1", PresetType::PRESET_P1)
        .value("P2", PresetType::PRESET_P2)
        .value("P3", PresetType::PRESET_P3)
        .value("P4", PresetType::PRESET_P4)
        .value("P5", PresetType::PRESET_P5)
        .value("P6", PresetType::PRESET_P6)
        .value("P7", PresetType::PRESET_P7);

    py::enum_<TuningInfo>(m, "TuningInfo")
        .value("HighQuality", TuningInfo::TUNING_INFO_HIGH_QUALITY)
        .value("LowLatency", TuningInfo::TUNING_INFO_LOW_LATENCY)
        .value("UltraLowLatency", TuningInfo::TUNING_INFO_ULTRA_LOW_LATENCY)
        .value("Lossless", TuningInfo::TUNING_INFO_LOSSLESS);

    py::class_<EncodeQp>(m, "EncodeQp")
        .def(py::init<>())
        .def_readwrite("qpInterP", &EncodeQp::qpInterP)
        .def_readwrite("qpInterB", &EncodeQp::qpInterB)
        .def_readwrite("qpIntra", &EncodeQp::qpIntra);

    py::class_<TensorEncodeConfig>(m, "TensorEncodeConfig")
        .def(py::init<>())
        .def_readwrite("input_format", &TensorEncodeConfig::input_format)
        .def_readwrite("codec_type", &TensorEncodeConfig::codec_type)
        .def_readwrite("rc_mode", &TensorEncodeConfig::rc_mode)
        .def_readwrite("preset", &TensorEncodeConfig::preset)
        .def_readwrite("tuning_info", &TensorEncodeConfig::tuning_info)
        .def_readwrite("average_bit_rate", &TensorEncodeConfig::average_bit_rate)
        .def_readwrite("max_bit_rate", &TensorEncodeConfig::max_bit_rate)
        .def_readwrite("target_quality", &TensorEncodeConfig::target_quality)
        .def_readwrite("const_qp", &TensorEncodeConfig::const_qp)
        .def_readwrite("spatial_aq", &TensorEncodeConfig::spatial_aq)
        .def_readwrite("temporal_aq", &TensorEncodeConfig::temporal_aq)
        .def_readwrite("monochrome", &TensorEncodeConfig::monochrome)
        .def_readwrite("gop_length", &TensorEncodeConfig::gop_length)
        .def_readwrite("frame_interval_p", &TensorEncodeConfig::frame_interval_p);

    py::class_<TensorEncodeOutput>(m, "TensorEncodeOutput")
        .def(py::init<>())
        .def_readonly("bitstream", &TensorEncodeOutput::bitstream)
        .def_readonly("packet_sizes", &TensorEncodeOutput::packet_sizes);

    py::class_<TensorEncoder>(m, "TensorEncoder")
        .def(py::init<const TensorEncodeConfig&, uint32_t, uint32_t>())
        .def("close", &TensorEncoder::close)
        .def("encode", &TensorEncoder::encode);

    py::class_<TensorDecoder>(m, "TensorDecoder")
        .def(py::init<CodecType, int, int>())
        .def("close", &TensorDecoder::close)
        .def("decode", &TensorDecoder::decode);
}
