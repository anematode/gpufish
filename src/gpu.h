
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
            uint32_t h = head;
            uint32_t next = (h + 1) % InstructionQueueSize;
            while (tail == next)
            {
                asm("pause");
            }

            // TODO block until there's space to submit
            queue[h] = instr;
            head = next;
        }

        std::array<int16_t, L1Size> read_scratch(size_t index);

        void blockUntilFinished() const
        {
            while (head != tail)
            {
                asm ("pause");
            }
        }

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

        Instruction queue[InstructionQueueSize];
        alignas(64) volatile uint32_t head;
        alignas(64) volatile uint32_t tail;

        alignas(64) int32_t result[16];

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