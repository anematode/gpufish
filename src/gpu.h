
#ifndef GPUFISH_GPU_H
#define GPUFISH_GPU_H
#include <array>
#include <memory>
#include <mutex>

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
        void init();
        void deinit();

        void submit(Instruction instr);

        void flush();
        void blockUntilComplete();
        bool ready() const;

        std::array<int16_t, L1Size> read_scratch(size_t index);

        template<Eval::NNUE::SIMD::UpdateOperation... ops,
                 std::enable_if_t<sizeof...(ops) == 0, bool> = true>
        void update_features([[maybe_unused]] Reg reg) {}

        template<Eval::NNUE::SIMD::UpdateOperation update_op,
                 Eval::NNUE::SIMD::UpdateOperation... ops,
                 typename T,
                 typename... Ts>
        void update_features(Reg reg, T index, Ts... indices)
        {
            submit(update_op == Eval::NNUE::SIMD::Add ? Instruction::add_feature(reg, index) : Instruction::sub_feature(reg, index));
            update_features<ops...>(reg, indices...);
        }

        bool isActive;

        void* stream;
        uint32_t queueIndex;

        alignas(64) volatile uint32_t instructionCount;
        Instruction queue[MaxInstructionsCount];

        // Result is written here by GPU. So that we keep the transfer to 64 bytes, we repurpose
        // result[i] == INT_MIN to mean "not (yet) written", and rely on 4-byte stores (at least) to
        // be atomic.
        alignas(64) volatile int32_t result[16];

        // Shared weights
        WeightsData *weights;

        // Device-side data pointer
        RegisterData *data;
    };


    class CudaContext
    {
        void *stream = nullptr;
        std::mutex streamCreationMtx;

    public:
        RegisterMachine *machines;
        size_t machineCount;
        std::unique_ptr<WeightsData> weights;

        CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount);
        void stop_all();
        RegisterMachine* get_machine(size_t size);

        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        void launch_persistent_kernel();
        ~CudaContext();
    };

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count);

}


#endif //GPUFISH_GPU_H