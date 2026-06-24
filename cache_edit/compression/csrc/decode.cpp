#include <memory>
#include <vector>
#include <cassert>
#include <cstddef>
#include "nvcodec.h"

static const int CODEC_SIZE = 2;
static const cudaVideoCodec vCodec[CODEC_SIZE] = {
    cudaVideoCodec_H264,
    cudaVideoCodec_HEVC,
};

TensorDecoder::TensorDecoder(CodecType codec_type, int maxWidth, int maxHeight) {
    CUcontext cuContext = NULL;
    // use cuDevicePrimaryCtxRetain?
    CUresult res = cuCtxGetCurrent(&cuContext);
    if (res != CUDA_SUCCESS || cuContext == NULL) {
        throw std::runtime_error("PyTorch CUDA context is not initialized");
    }
    torch_cu_context = cuContext;

    cudaVideoCodec codec = vCodec[static_cast<int>(codec_type)];
    nv_decoder.reset(new NvDecoder(
        cuContext,
        true,
        codec,
        false,
        false,
        nullptr,
        nullptr,
        false,
        maxWidth,
        maxHeight
    ));
}

TensorDecoder::~TensorDecoder() {
    close();
}

void TensorDecoder::close() {
    if (nv_decoder) {
        nv_decoder.reset();
    }
}

torch::Tensor TensorDecoder::decode(torch::Tensor bitstream, torch::Tensor packet_sizes) {
    bitstream = bitstream.to(torch::kCPU);
    packet_sizes = packet_sizes.to(torch::kCPU);

    int64_t* packet_sizes_ptr = packet_sizes.data_ptr<int64_t>();
    uint8_t* bitstream_ptr = bitstream.data_ptr<uint8_t>();
    size_t bitstream_offset = 0;

    auto output_options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output_tensor = torch::empty({0}, output_options);
    int frame_size = 0;
    size_t output_tensor_offset = 0;
    nv_decoder->setReconfigParams(nullptr, nullptr);

    cudaVideoSurfaceFormat output_format;
    int num_packets = packet_sizes.size(0);
    int num_frames = 0;
    for (int i = 0; i < num_packets + 1; i++) {
        int packet_size;
        if (i < num_packets) {
            packet_size = static_cast<int>(*(packet_sizes_ptr + i));
        } else {
            packet_size = 0;
        }
        int num_frames_return = nv_decoder->Decode(bitstream_ptr + bitstream_offset, packet_size);
        bitstream_offset += static_cast<size_t>(packet_size);
        if (num_frames_return == 0) {
            continue;
        }

        if (frame_size == 0) {
            output_format = nv_decoder->GetOutputFormat();
            if (output_format != cudaVideoSurfaceFormat_YUV444 && output_format != cudaVideoSurfaceFormat_NV12) {
                throw std::runtime_error("Only YUV444 or NV12 output format is supported");
            }
            frame_size = nv_decoder->GetFrameSize();
            output_tensor.resize_({static_cast<int64_t>(frame_size) * static_cast<int64_t>(num_packets)});
        }
        assert(nv_decoder->GetWidth() == nv_decoder->GetDecodeWidth());
        for (int frame_idx = 0; frame_idx < num_frames_return; frame_idx++) {
            if (output_tensor_offset + frame_size > output_tensor.numel()) {
                output_tensor.resize_({output_tensor.numel() * 2});
            }
            uint8_t* p_frame = nv_decoder->GetFrame();
            ck(cuMemcpyDtoD(
                (CUdeviceptr)(output_tensor.data_ptr<uint8_t>() + output_tensor_offset),
                (CUdeviceptr)p_frame,
                frame_size
            ));
            // P-frame GOPs may return several decoded frames in one Decode()
            // call, especially on flush. Each frame needs its own output slot.
            output_tensor_offset += frame_size;
            num_frames++;
        }
    }

    output_tensor.resize_({static_cast<int64_t>(frame_size) * static_cast<int64_t>(num_frames)});
    if (output_format == cudaVideoSurfaceFormat_YUV444) {
        output_tensor = output_tensor.reshape({num_frames, 3, nv_decoder->GetHeight(), nv_decoder->GetWidth()});
    } else {
        int luma_height = nv_decoder->GetHeight();
        int total_height = luma_height + (luma_height + 1) / 2;
        assert(total_height == (luma_height + nv_decoder->GetChromaHeight() * nv_decoder->GetNumChromaPlanes()));
        int luma_width = nv_decoder->GetWidth();
        output_tensor = output_tensor.reshape({num_frames, total_height, luma_width});
    }
    return output_tensor;
}
