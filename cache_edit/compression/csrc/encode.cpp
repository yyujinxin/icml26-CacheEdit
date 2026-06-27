#include <memory>
#include <vector>
#include <exception>
#include <cstddef>
#include <cassert>
#include "c10/cuda/CUDAStream.h"
#include "nvcodec.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvCodecUtils.h"
#include "Logger.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

static const int CODEC_SIZE = 2;
static const GUID vCodec[CODEC_SIZE] = {
    NV_ENC_CODEC_H264_GUID,
    NV_ENC_CODEC_HEVC_GUID,
};

static const int PRESET_SIZE = 7;
static const GUID vPreset[PRESET_SIZE] = {
    NV_ENC_PRESET_P1_GUID,
    NV_ENC_PRESET_P2_GUID,
    NV_ENC_PRESET_P3_GUID,
    NV_ENC_PRESET_P4_GUID,
    NV_ENC_PRESET_P5_GUID,
    NV_ENC_PRESET_P6_GUID,
    NV_ENC_PRESET_P7_GUID,
};

static const int TUNING_SIZE = 4;
static const NV_ENC_TUNING_INFO vTuningInfo[TUNING_SIZE] = {
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
    NV_ENC_TUNING_INFO_LOW_LATENCY,
    NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
    NV_ENC_TUNING_INFO_LOSSLESS
};


static const int RC_MODE_SIZE = 3;
static const NV_ENC_PARAMS_RC_MODE vRcMode[RC_MODE_SIZE] = {
    NV_ENC_PARAMS_RC_CONSTQP,
    NV_ENC_PARAMS_RC_VBR,
    NV_ENC_PARAMS_RC_CBR,
};

static const int INPUT_FORMAT_SIZE = 3;
static const NV_ENC_BUFFER_FORMAT vInputFormat[INPUT_FORMAT_SIZE] = {
    NV_ENC_BUFFER_FORMAT_YUV444,
    NV_ENC_BUFFER_FORMAT_ARGB,
    NV_ENC_BUFFER_FORMAT_NV12,
};

TensorEncoder::TensorEncoder(const TensorEncodeConfig& config, uint32_t maxWidth, uint32_t maxHeight) {
    CUcontext cuContext = NULL;
    // use cuDevicePrimaryCtxRetain?
    CUresult res = cuCtxGetCurrent(&cuContext);
    if (res != CUDA_SUCCESS || cuContext == NULL) {
        throw std::runtime_error("PyTorch CUDA context is not initialized");
    }

    // use default CUDA stream getDefaultCUDAStream?
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    torch_cu_context = cuContext;
    torch_cu_stream = torch_stream.stream();

    buffer_format = vInputFormat[static_cast<int>(config.input_format)];
    nv_encoder.reset(new NvEncoderOutputInVidMemCuda(
        cuContext,
        maxWidth,
        maxHeight,
        buffer_format,
        false
    ));
	NV_ENC_INITIALIZE_PARAMS initialize_params = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encode_config = { NV_ENC_CONFIG_VER };

    GUID encode_guid = vCodec[static_cast<int>(config.codec_type)];
    GUID preset_guid = vPreset[static_cast<int>(config.preset)];
    NV_ENC_TUNING_INFO tuning_info = vTuningInfo[static_cast<int>(config.tuning_info)];
	initialize_params.encodeConfig = &encode_config;
	nv_encoder->CreateDefaultEncoderParams(
        &initialize_params,
        encode_guid,
        preset_guid,
        tuning_info
    );

    NV_ENC_PARAMS_RC_MODE rc_mode = vRcMode[static_cast<int>(config.rc_mode)];
    encode_config.rcParams.rateControlMode = rc_mode;

    if (config.monochrome.has_value() && config.monochrome.value()) {
        if (buffer_format == NV_ENC_BUFFER_FORMAT_NV12) {
            encode_config.monoChromeEncoding = 1;
            monochrome = true;
        } else {
            throw std::invalid_argument("monochrome encoding is only supported for NV12 input format");
        }
    } else {
        if (buffer_format == NV_ENC_BUFFER_FORMAT_NV12) {
            throw std::invalid_argument("NV12 input format is currently only supported for monochrome encoding");
        }
        monochrome = false;
    }

    if (rc_mode == NV_ENC_PARAMS_RC_CONSTQP) {
        if (!(config.const_qp.has_value())) {
            throw std::invalid_argument("const_qp must be set for constant QP mode");
        }
    }
    if ((rc_mode == NV_ENC_PARAMS_RC_VBR && !config.target_quality.has_value()) || rc_mode == NV_ENC_PARAMS_RC_CBR) {
        if (!(config.average_bit_rate.has_value())) {
            throw std::invalid_argument("average_bit_rate must be set for VBR or CBR mode");
        }
    }
    if (config.average_bit_rate.has_value()) {
        encode_config.rcParams.averageBitRate = config.max_bit_rate.value();
    }
    if (config.max_bit_rate.has_value()) {
        encode_config.rcParams.maxBitRate = config.max_bit_rate.value();
    }
    if (config.target_quality.has_value()) {
        encode_config.rcParams.targetQuality = config.target_quality.value();
    }
    if (config.spatial_aq.has_value()) {
        encode_config.rcParams.enableAQ = true;
        encode_config.rcParams.aqStrength = config.spatial_aq.value();
    }
    if (config.temporal_aq.has_value()) {
        encode_config.rcParams.enableTemporalAQ = config.temporal_aq.value();
    }

    if (config.const_qp.has_value()) {
        const EncodeQp& encode_qp = config.const_qp.value();
        encode_config.rcParams.constQP.qpInterP = encode_qp.qpInterP;
        encode_config.rcParams.constQP.qpInterB = encode_qp.qpInterB;
        encode_config.rcParams.constQP.qpIntra = encode_qp.qpIntra;
    }

    // GOP configuration for inter-frame prediction
    if (config.gop_length.has_value()) {
        encode_config.gopLength = config.gop_length.value();
    }
    if (config.frame_interval_p.has_value()) {
        encode_config.frameIntervalP = config.frame_interval_p.value();
    }

	nv_encoder->CreateEncoder(&initialize_params);
    nv_encoder->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&torch_cu_stream, (NV_ENC_CUSTREAM_PTR)&torch_cu_stream);

    ck(cuMemAllocHost((void **)&output_params_host, sizeof(NV_ENC_ENCODE_OUT_PARAMS)));
}

TensorEncoder::~TensorEncoder() {
    close();
}

void TensorEncoder::close() {
    if (nv_encoder) {
        nv_encoder.reset();
    }
    if (output_params_host != nullptr) {
        cuMemFreeHost(output_params_host);
        output_params_host = nullptr;
    }
}

TensorEncodeOutput TensorEncoder::encode(torch::Tensor input) {
    if (input.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("tensor encode input must be int8 tensor");
    }
    int target_width;
    int target_height;
    if (monochrome) {
        if (buffer_format == NV_ENC_BUFFER_FORMAT_NV12) {
            if (input.dim() != 3) {
                throw std::invalid_argument("tensor encode input must be 3D tensor for monochrome encoding");
            }
            target_width = input.size(-1);
            int total_height = input.size(-2);
            // luma height
            target_height = total_height * 2 / 3;
        } else {
            throw std::invalid_argument("unsupported input format for monochrome encoding");
        }
    } else {
        if (input.dim() != 4) {
            throw std::invalid_argument("tensor encode input must be 4D tensor for colour encoding");
        }
        if (buffer_format == NV_ENC_BUFFER_FORMAT_YUV444) {
            target_width = input.size(-1);
            target_height = input.size(-2);
            if (input.size(1) != 3) {
                throw std::invalid_argument("tensor encode input must have 3 channels (YUV444)");
            }
        } else if (buffer_format == NV_ENC_BUFFER_FORMAT_ARGB) {
            target_height = input.size(1);
            target_width = input.size(2);
            if (input.size(-1) != 4) {
                // ARGB input format for NVENC
                // is 8 bit Packed A8R8G8B8. This is a word-ordered format
                // where a pixel is represented by a 32-bit word with B
                // in the lowest 8 bits, G in the next 8 bits, R in the
                // 8 bits after that and A in the highest 8 bits.
                // NVIDIA GPUs are little-endian
                // so the last dim follows the order of B, G, R, A
                throw std::invalid_argument("tensor encode input must have 4 channels (ARGB)");
            }
        }  else {
            throw std::invalid_argument("unsupported input format for colour encoding");
        }
    }

    int total_frames = input.size(0);
    int encoder_width = nv_encoder->GetEncodeWidth();
    int encoder_height = nv_encoder->GetEncodeHeight();

    NV_ENC_RECONFIGURE_PARAMS reconfigureParams = {NV_ENC_RECONFIGURE_PARAMS_VER};
    reconfigureParams.reInitEncodeParams = nv_encoder->GetinitializeParams();
    reconfigureParams.reInitEncodeParams.encodeWidth = target_width;
    reconfigureParams.reInitEncodeParams.encodeHeight = target_height;
    // This is necessary to reset the internal encoder state for new stream
    reconfigureParams.resetEncoder = 1;
    nv_encoder->Reconfigure(&reconfigureParams);

	int frame_size = nv_encoder->GetFrameSize();
    if (buffer_format == NV_ENC_BUFFER_FORMAT_YUV444) {
        assert(frame_size == target_width * target_height * 3);
    } else if (buffer_format == NV_ENC_BUFFER_FORMAT_ARGB) {
        assert(frame_size == target_width * target_height * 4);
    } else if (buffer_format == NV_ENC_BUFFER_FORMAT_NV12) {
        assert(frame_size == target_width * (target_height + (target_height + 1) / 2));
    }

    int num_frames_encoded = 0;
    uint8_t* data_ptr = input.data_ptr<uint8_t>();
    ptrdiff_t data_offset = 0;

    auto output_options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    // TODO(yongji): initialize output_bitstream with a size that is a fraction of the raw input size
    // according to e.g., the average compression ratio
    int64_t raw_size = static_cast<int64_t>(frame_size) * static_cast<int64_t>(total_frames);
    torch::Tensor output_bitstream = torch::empty({raw_size}, output_options);
    std::vector<int64_t> output_bs_sizes;
    output_bs_sizes.reserve(total_frames);
    uint64_t output_bitstream_offset = 0;
    uint64_t num_packets = 0;
    while (num_frames_encoded <= total_frames) {
		std::vector<NV_ENC_OUTPUT_PTR> output_video_mem_buffer;
        if (num_frames_encoded < total_frames) {
            const NvEncInputFrame* encoder_input_frame = nv_encoder->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(
                torch_cu_context,
                (void*)data_ptr + data_offset,
                0,
                (CUdeviceptr)encoder_input_frame->inputPtr,
                (int)encoder_input_frame->pitch,
                nv_encoder->GetEncodeWidth(),
                nv_encoder->GetEncodeHeight(),
                CU_MEMORYTYPE_DEVICE,
                encoder_input_frame->bufferFormat,
                encoder_input_frame->chromaOffsets,
                encoder_input_frame->numChromaPlanes,
                false,
                NULL
            );
            nv_encoder->EncodeFrame(output_video_mem_buffer);
        } else {
            nv_encoder->EndEncode(output_video_mem_buffer);
        }
        num_frames_encoded++;

        ck(cuCtxPushCurrent(torch_cu_context));
        for (uint32_t i = 0; i < output_video_mem_buffer.size(); i++) {
            // Copy encoded frame from video memory buffer to host memory buffer
            ck(cuMemcpyDtoH(
                (void*)output_params_host,
                (CUdeviceptr)output_video_mem_buffer[i],
                sizeof(NV_ENC_ENCODE_OUT_PARAMS)
            ));
            uint32_t offset = sizeof(NV_ENC_ENCODE_OUT_PARAMS);
            uint32_t bitstream_size = output_params_host->bitstreamSizeInBytes;
            if (output_bitstream_offset + bitstream_size > output_bitstream.numel()) {
                output_bitstream.resize_({output_bitstream.numel() * 2});
            }
            ck(cuMemcpyDtoD(
                (CUdeviceptr)((void*)output_bitstream.data_ptr<uint8_t>() + output_bitstream_offset),
                (CUdeviceptr)(output_video_mem_buffer[i] + offset),
                bitstream_size
            ));
            output_bitstream_offset += bitstream_size;
            output_bs_sizes.push_back(bitstream_size);
            num_packets++;
        }
        ck(cuCtxPopCurrent(NULL));
        data_offset += frame_size;
    }

    // Build packet_sizes tensor directly on GPU (avoiding CPU->GPU copy)
    auto bs_sizes_options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor bs_sizes_tensor = torch::empty({static_cast<int64_t>(num_packets)}, bs_sizes_options);
    // Copy the vector data to GPU tensor
    ck(cuMemcpyHtoD(
        (CUdeviceptr)bs_sizes_tensor.data_ptr<int64_t>(),
        output_bs_sizes.data(),
        num_packets * sizeof(int64_t)
    ));
    output_bitstream.resize_({output_bitstream_offset});

    // Move both bitstream and packet_sizes to CPU (pinned memory) asynchronously
    // so they're ready for NVDEC on decode. This avoids redundant D->H copies on
    // every cache reuse. Use pinned memory + non_blocking=true to overlap the
    // transfer with subsequent computation.
    torch::Tensor bitstream_cpu = torch::empty(
        {output_bitstream.size(0)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true)
    );
    torch::Tensor bs_sizes_cpu = torch::empty(
        {bs_sizes_tensor.size(0)},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true)
    );
    // non_blocking=true: D->H transfer happens asynchronously on the current CUDA
    // stream. PyTorch will automatically insert stream synchronization when the
    // CPU tensor is accessed, so the caller doesn't need explicit synchronization.
    bitstream_cpu.copy_(output_bitstream, /*non_blocking=*/true);
    bs_sizes_cpu.copy_(bs_sizes_tensor, /*non_blocking=*/true);

    return TensorEncodeOutput {
        bitstream_cpu,
        bs_sizes_cpu
    };
}
