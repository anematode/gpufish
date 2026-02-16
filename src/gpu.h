
#ifndef GPUFISH_GPU_H
#define GPUFISH_GPU_H
#include <array>
#include <memory>
#include <mutex>
#include <climits>

#include "gpu_defs.h"
#include "nnue/network.h"

namespace Stockfish::GPU
{
    // Forward decls
    struct WeightsData;
    struct RegisterData;

    // May be allocated in pinned host WC memory; instructions are copied here from the staging buffer. The GPU
    // polls the instructionCount to know when to start stepping. The instruction count should always be
    // written last (and technically with a store fence, but we're just using volatile and TSO for now).
    struct alignas(64) WCInstructionBuffer
    {
        union
        {
            struct
            {
                uint16_t instructionCount;
                uint16_t id;
            } s;
            uint32_t data;
        };
        Instruction list[MaxInstructionsCount];
        char padding[64];

        void flush(WCInstructionBuffer* to)
        {
            s.id++;

            uint32_t count = s.instructionCount;
            constexpr bool UseMovdir64B = false;
            if constexpr (UseMovdir64B)
            {
                char* dest = reinterpret_cast<char*>(to);
                const char* src = reinterpret_cast<char*>(this);

                // We need to copy this many lines in reverse
                ptrdiff_t lines = (count * sizeof(Instruction) + sizeof(s.instructionCount) + 63) / 64;
                for (ptrdiff_t j = lines - 1; j >= 0; --j)
                {
                    asm ("movdir64b %1, %0" :: "r"(dest + 64 * j), "m"(src[64 * j]) : "memory");
                }
            } else
            {
                memcpy(&to->list, list, sizeof(Instruction) * count);
                asm volatile ("" ::: "memory" );
                memcpy(&to->data, &data, 4);
                asm ("sfence");
            }
        }
    };

    // Allocated on the host in pinned memory
    struct RegisterMachine
    {
        void init();
        void deinit();
        std::array<int16_t, L1Size> read_scratch(size_t index);

        void flush()
        {
            if (staging.s.instructionCount == 0)
            {
                result[0] = 0;  // prevent accidentally waiting
                return;
            }

            std::fill_n(result, 16, INT_MIN);
            staging.flush(wcBuffer);
            std::fill_n(regStates, 4, -1);
        }

        void blockUntilComplete()
        {
            while (!ready())  // TODO add a "perf counter" for this
            {
                asm("pause");
                /*if (attempts++ >= 1000000)
                {
                    std::cout << "Register machine at " << this << " failed to read the result in time!\n";
                    for (int i = 0; i < staging.instructionCount; i++)
                    {
                        std::cout << "Instruction " << i << " " << staging.list[i].to_string() << "\n";
                    }
                    abort();
                }*/
            }

            staging.s.instructionCount = 0;
        }

        bool ready() const
        {
            return result[0] != INT_MIN;
        }

        __attribute__((always_inline)) void submit(Instruction instr)
        {
#ifndef NDEBUG
            if (!isActive)
            {
                fprintf(stderr, "RegisterMachine is inactive!\n");
                abort();
            }
#endif

            if (staging.s.instructionCount >= MaxInstructionsCount)
            {
                // Need an immediate flush before writing the next instruction
                // Mainly used during setup
                flush();
                blockUntilComplete();
            }

            switch (instr.opcode())
            {
            case LdScratch:
                if (regStates[instr.decode_reg()] == int(instr.decode_wide_index()))
                {
                    return;
                }
                break;
            case StScratch:
                regStates[instr.decode_reg()] = int(instr.decode_wide_index());
                break;
            case AddFeature:
            case SubFeature:
            case ResetReg:
                regStates[instr.decode_reg()] = -1;
                break;
            default: break;
            }

            staging.list[staging.s.instructionCount++] = instr;
        }


        std::array<int32_t, 16> read_result() const
        {
            std::array<int32_t, 16> r;
            std::copy_n(this->result, 16, r.begin());
            return r;
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

        bool isActive;

        void* stream;

        WCInstructionBuffer* wcBuffer;
        WCInstructionBuffer staging;

        // Result is written here by GPU. So that we keep the transfer to 64 bytes, we repurpose
        // result[i] == INT_MIN to mean "not (yet) written", and rely on 4-byte stores (at least) to
        // be atomic.
        alignas(64) volatile int32_t result[16];

        // Shared weights
        WeightsData *weights;

        // Device-side data pointer
        RegisterData *data;

        // Scratch index that this register is equal to, or -1
        int regStates[4];
    };


    class CudaContext
    {
        void *stream = nullptr;
        std::mutex streamCreationMtx;

    public:
        RegisterMachine *machines;
        WCInstructionBuffer* wcBuffers;
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