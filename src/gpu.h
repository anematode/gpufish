
#ifndef GPUFISH_GPU_H
#define GPUFISH_GPU_H
#include <array>
#include <memory>

#include "gpu_defs.h"
#include "nnue/network.h"

namespace Stockfish::GPU
{
    // Forward decls
    struct WeightsData;
    struct RegisterData;

    // Allocated on the host in pinned memory
    struct RegisterMachine
    {
        void submit(Instruction instr)
        {
            queue[head++] = instr;
            head %= InstructionQueueSize;
        }

        Instruction queue[InstructionQueueSize];
        alignas(64) volatile uint32_t head;
        alignas(64) volatile uint32_t tail;

        // Shared weights
        WeightsData *weights;

        // device-side data pointer
        RegisterData *data;
    };


    class CudaContext
    {
        void *stream = nullptr;

    public:
        RegisterMachine *machines;
        size_t machineCount;
        std::unique_ptr<WeightsData> weights;

        CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount);

        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        void launch_persistent_kernel();
        ~CudaContext();
    };

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count);

}


#endif //GPUFISH_GPU_H