#ifndef _TENSOR_NVENC_H
#define _TENSOR_NVENC_H

#include <cstdint>
#include <memory>
#include <optional>
#include <torch/extension.h>
#include <cuda.h>
#include "NvEncoder/NvEncoderOutputInVidMemCuda.h"
#include "NvDecoder/NvDecoder.h"


enum class InputFormat {
    INPUT_YUV444 = 0,
    INPUT_ARGB = 1,
    INPUT_NV12 = 2,
};

enum class CodecType {
    CODEC_H264 = 0,
    CODEC_HEVC = 1,
};

enum class RateControlMode {
    RC_CONSTQP = 0,
    RC_VBR = 1,
    RC_CBR = 2,
};

enum class PresetType {
    PRESET_P1 = 0,
    PRESET_P2 = 1,
    PRESET_P3 = 2,
    PRESET_P4 = 3,
    PRESET_P5 = 4,
    PRESET_P6 = 5,
    PRESET_P7 = 6,
};

enum class TuningInfo {
    TUNING_INFO_HIGH_QUALITY = 0 ,
    TUNING_INFO_LOW_LATENCY = 1,
    TUNING_INFO_ULTRA_LOW_LATENCY = 2,
    TUNING_INFO_LOSSLESS = 3,
};


struct EncodeQp {
    uint32_t qpInterP;
    uint32_t qpInterB;
    uint32_t qpIntra;
};

struct TensorEncodeConfig {
    InputFormat input_format;

    CodecType codec_type;
    RateControlMode rc_mode;
    PresetType preset;
    TuningInfo tuning_info;

    std::optional<uint32_t> average_bit_rate;
    std::optional<uint32_t> max_bit_rate;
    std::optional<uint8_t> target_quality;
    std::optional<EncodeQp> const_qp;
    std::optional<int> spatial_aq;
    std::optional<bool> temporal_aq;
    std::optional<bool> monochrome;

    // GOP configuration for inter-frame prediction
    std::optional<uint32_t> gop_length;        // GOP length (frames between I-frames)
    std::optional<uint32_t> frame_interval_p;  // P-frame interval (1=IPPP, 2=IBPBP, 3=IBBPBBP)
};

struct TensorEncodeOutput {
    torch::Tensor bitstream;
    torch::Tensor packet_sizes;
};

class TensorEncoder {
public:
    TensorEncoder(const TensorEncodeConfig& config, uint32_t maxWidth, uint32_t maxHeight);
    ~TensorEncoder();
    void close();
    TensorEncodeOutput encode(torch::Tensor input);
private:
    std::unique_ptr<NvEncoderOutputInVidMemCuda> nv_encoder;
    CUcontext torch_cu_context;
    CUstream torch_cu_stream;
    NV_ENC_ENCODE_OUT_PARAMS *output_params_host = nullptr;
    NV_ENC_BUFFER_FORMAT buffer_format;
    bool monochrome;
};

class TensorDecoder {
public:
    TensorDecoder(CodecType codec_type, int maxWidth, int maxHeight);
    ~TensorDecoder();
    void close();
    torch::Tensor decode(torch::Tensor bitstream, torch::Tensor packet_sizes);
private:
    std::unique_ptr<NvDecoder> nv_decoder;
    CUcontext torch_cu_context;
};
#endif
