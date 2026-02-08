
#ifndef GPUFISH_GPU_H
#define GPUFISH_GPU_H
#include <array>
#include <memory>

#include "gpu_defs.h"
#include "nnue/network.h"


namespace Stockfish::GPU
{
    struct RegisterMachine;
    struct WeightsData;

    class CudaContext
    {
    public:
        RegisterMachine *machines;
        size_t machineCount;
        std::unique_ptr<WeightsData> weights;

        CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount);

        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        ~CudaContext();
    };

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count);

    // Per-thread accumulator stack instantiation that accepts update commands, which a GPU kernel will consume.
    class AsyncAccStack {
        static constexpr size_t MaxInstructionCount = 256;
    public:
        AsyncAccStack(RegisterMachine* ctx);
        ~AsyncAccStack();

        void submit_instruction(Instruction instr);
        void commit();

        [[nodiscard]] bool poll() const;
        [[nodiscard]] std::array<int32_t, 16> read_l2_result() const;

    private:
        Instruction instrs[MaxInstructionCount] = {};

        // Result obtained from GPU, will be piped into the subsequent layers
        alignas(64) std::array<int32_t, 16> l1_result;
        RegisterMachine* machine = nullptr;
    };
}


#endif //GPUFISH_GPU_H