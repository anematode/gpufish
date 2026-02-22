
#ifndef GPUFISH_GPU_H
#define GPUFISH_GPU_H
#include <array>
#include <memory>
#include <mutex>
#include <climits>

#include "gpu_defs.h"
#include "nnue/network.h"

namespace Stockfish::GPU {
// (Forward decls)
struct WeightsData;
struct RegisterData;

// May be allocated in pinned host WC memory; instructions are copied CPU-side from the staging buffer. The GPU
// polls the instructionCount to know when to start stepping. The instruction count should always be
// written last (and technically with a store fence, but we're just using volatile and TSO for now).
struct alignas(64) InstructionBuffer {
    // Low 16 bits are the instruction count; upper 16 bits are an id that serve to tell the GPU
    // that the buffer has been updated.
    uint32_t    data;
    Instruction list[MaxInstructionsCount];
    char        padding[64];

    int get_instruction_count() const {
        uint16_t val;
        memcpy(&val, &data, 2);
        return val;
    }

    void set_instruction_count(uint16_t count) {
        assert(get_instruction_count() < MaxInstructionsCount - 1);
        memcpy(&data, &count, 2);
    }


    void flush(InstructionBuffer* to) {
        memcpy(&to->list, list, sizeof(Instruction) * get_instruction_count());
        // write to instruction count must occur after
        std::atomic_thread_fence(std::memory_order_release);

        // increment ID so GPU can distinguish two payloads with equal instruction counts
        data += 0x10000;
        to->data = data;

        // The destination buffer is allocated in write-combining memory, so we use an sfence to flush the WC queue
        // (i.e. this is for perf and not correctness)
#ifdef __x86_64__
        asm("sfence");
#endif
    }
};

// Allocated on the host in pinned memory (allowing efficient DMA from the GPU).
// Important functions are implemented here in the header to allow inlining (the .cu code can't be LTOed).
struct RegisterMachine {
    RegisterMachine(const WeightsData* weights, InstructionBuffer* wcInstructionBuffer);
    ~RegisterMachine();

    void flush() {
        if (staging.get_instruction_count() == 0)
        {
            result[0] = result[16] = 0;  // prevent accidentally blocking with blockUntilComplete()
            return;
        }

        std::fill_n(result, 32, INT_MIN);
        staging.flush(wcBuffer);
        std::fill_n(regStates, 4, -1);
    }

    void blockUntilComplete() {
        while (!ready())
        {
#ifdef __x86_64__
            asm("pause");
#endif
        }

        staging.set_instruction_count(0);
    }

    [[nodiscard]] bool ready() const
    {
        return result[0] != INT_MIN && result[16] != INT_MIN;
    }

    // Marked always inline because we usually call with a constant instruction type, which allows
    // folding of the switc
    __attribute__((always_inline)) void submit(Instruction instr) {
        assert(isActive && "Register machine is inactive");

        int instrCount = staging.get_instruction_count();
        if (__builtin_expect(instrCount >= MaxInstructionsCount - 1, 0))
        {
            // Need an immediate flush before writing the next instruction.
            // Mainly used during setup, but important for correctness.
            flush();
            blockUntilComplete();
            instrCount = 0;
        }

        switch (instr.opcode())
        {
        case LdScratch :
            if (regStates[instr.decode_reg()] == int(instr.decode_wide_index()))
            {
                // We're loading from a scratch reg that we just wrote this register to -- skip it
                return;
            }
            break;
        case StScratch :
            regStates[instr.decode_reg()] = int(instr.decode_wide_index());
            break;
        case AddFeature :
        case SubFeature :
        case ResetReg :
            regStates[instr.decode_reg()] = -1;
            break;
        default :
            break;
        }

        staging.list[instrCount++] = instr;
        staging.set_instruction_count(instrCount);
    }

    Instruction& peek() {
        assert(staging.get_instruction_count() > 0);
        return staging.list[staging.get_instruction_count() - 1];
    }

    std::array<int32_t, 32> read_result() const {
        std::array<int32_t, 32> r;
        memcpy(&r, const_cast<const int32_t*>(result), sizeof(r));
        return r;
    }

    template<Eval::NNUE::SIMD::UpdateOperation... ops,
             std::enable_if_t<sizeof...(ops) == 0, bool> = true>
    void update_features([[maybe_unused]] Reg reg) {}

    template<Eval::NNUE::SIMD::UpdateOperation update_op,
             Eval::NNUE::SIMD::UpdateOperation... ops,
             typename T,
             typename... Ts>
    void update_features(Reg reg, T index, Ts... indices) {
        submit(update_op == Eval::NNUE::SIMD::Add ? Instruction::add_feature(reg, index)
                                                  : Instruction::sub_feature(reg, index));
        update_features<ops...>(reg, indices...);
    }

    bool  isActive;
    void* stream;  // cudaStream_t

    // Instructions are first placed here...
    InstructionBuffer staging;
    // ... then copied into here, which is being polled continuously by the GPU.
    InstructionBuffer* wcBuffer;

    // Result is written here by GPU. So that we keep the transfer to 64 bytes, we repurpose
    // result[i] == INT_MIN to mean "not (yet) written", and rely on 4-byte stores (at least) to
    // be atomic.
    alignas(64) volatile int32_t result[32];

    // Shared weights
    const WeightsData* weights;

    // Device-side data pointer
    RegisterData* data;

    // Scratch index that this register (regA through D) is equal to, or -1
    int regStates[4];

   private:
    // For kernel debugging
    std::array<int16_t, L1Size> read_scratch(size_t index) const;
};


class CudaContext {
   public:
    CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount);
    void             stop_all();
    RegisterMachine* get_machine(size_t size);

    CudaContext(const CudaContext&)            = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    void launch_persistent_kernel();
    ~CudaContext();

   private:
    void*                        stream = nullptr;  // cudaStream_t
    RegisterMachine*             machines;
    InstructionBuffer*           wcBuffers;
    size_t                       machineCount;
    std::unique_ptr<WeightsData> weights;
};

}


#endif  //GPUFISH_GPU_H
