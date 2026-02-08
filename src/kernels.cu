#include "gpu.h"
#include "gpu_defs.h"

#include <cstdio>
#include <memory>

#include "nnue/network.h"

#define InstructionQueueSize 2048
#define CacheLineSize 64
#define ThreadsPerWarp 32

// Credit: https://stackoverflow.com/a/14038590
#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


namespace Stockfish::GPU
{

    constexpr int L1EntriesPerThreadSlice = L1Size / ThreadsPerWarp;
    constexpr int PtxRegsPerThreadSlice = L1EntriesPerThreadSlice / 2;  // each unsigned contains two 16-bit values

    struct ScratchReg
    {
        int16_t data[L1Size];  // TODO: psqt?
    };

    // Device-side data that lives only in GPU memory
    struct RegisterData
    {
        ScratchReg regs[ScratchRegCount];

        int16_t *get_scratch(Instruction inst)
        {
            return regs[inst.decode_wide_index()].data;
        }
    };

    struct WeightsData
    {
        // Device-side pointers
        Eval::NNUE::BigFeatureTransformer* transformer;
        Eval::NNUE::L1Bucket *buckets;

        WeightsData(const Eval::NNUE::NetworkBig &big)
        {
            const Eval::NNUE::BigFeatureTransformer& transformer = big.get_ft();
            auto sparse_input_buckets = big.get_input_buckets();

            auto temp = std::make_unique<Eval::NNUE::BigFeatureTransformer>(transformer);
            temp->unpermute_weights();

            checkError(cudaMalloc(&this->transformer, sizeof(transformer)));
            checkError(cudaMemcpy(this->transformer, &temp, sizeof(transformer), cudaMemcpyHostToDevice));

            size_t bc = sparse_input_buckets.size();
            checkError(cudaMalloc(&buckets, sizeof(*sparse_input_buckets[0]) * bc));

            for (size_t i = 0; i < bc; i++)
            {
                auto biases = sparse_input_buckets[i]->get_biases();
                auto weights = sparse_input_buckets[i]->get_weights();

                checkError(cudaMemcpy(&buckets[i].biases, biases.data(), sizeof(buckets[i].biases), cudaMemcpyHostToDevice));
                checkError(cudaMemcpy(&buckets[i].weights, weights.data(), sizeof(buckets[i].weights), cudaMemcpyHostToDevice));
            }
        }

        WeightsData(const WeightsData&) = delete;

        ~WeightsData()
        {
            checkError(cudaFree(transformer));
            transformer = nullptr;

            checkError(cudaFree(buckets));
            buckets = nullptr;
        }
    };

    // Allocated on the host in pinned memory
    struct RegisterMachine
    {
        Instruction queue[InstructionQueueSize];
        alignas(CacheLineSize) volatile uint32_t head;
        alignas(CacheLineSize) volatile uint32_t tail;

        // Shared weights
        WeightsData *weights;

        // device-side data pointer
        RegisterData *data;

        __device__ bool devicePoll() const
        {
            return head == tail;
        }

        void submit(const Instruction* start, size_t count)
        {
            memcpy(queue + head, start, count * sizeof(Instruction));
        }
    };

    __device__ void cvt8_to_16(uint32_t data, uint32_t *l, uint32_t *h)
    {
        uint32_t lo, hi;
        asm ("prmt.b32 %[lo],%[data],0,0x9180;\n"
             "prmt.b32 %[hi],%[data],0,0xb3a2;" :  [lo]"=r"(lo), [hi]"=r"(hi) : [data]"r"(data));
        *l = lo;
        *h = hi;
    }

    bool is_halfka_reg(Reg reg)
    {
        return reg == A || reg == B;
    }

    __global__ void persistent_kernel(RegisterMachine* machines, int num_machines) {
        unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerWarp;
        unsigned lane_id = threadIdx.x % ThreadsPerWarp;

        // Each warp picks a queue to monitor
        if (warp_id >= num_machines) return;

        RegisterMachine *machine = &machines[warp_id];
        RegisterData *data = machine->data;
        auto* transformer = machine->weights->transformer;
        auto* buckets = machine->weights->buckets;

        typedef unsigned reg_t[PtxRegsPerThreadSlice];
        reg_t regA, regB, regC, regD;

        uint32_t myL1Offset = L1EntriesPerThreadSlice * lane_id;

#define SWITCH_REG(X) switch (inst.decode_reg()) { \
    case 0: { X(regA); break; } \
    case 1: { X(regB); break; } \
    case 2: { X(regC); break; } \
    case 3: { X(regD); break; } \
    default: __builtin_unreachable(); \
    };

        while (true) {
            __shared__ Instruction current_cmd[64];
            Instruction& inst = current_cmd[threadIdx.x / 32];

            // Warp leader polls the queue
            if (lane_id == 0) {
                while (machine->devicePoll());
                // todo exit somehow lol
                inst = machine->queue[machine->tail];
            }

            uint32_t mask = __activemask();

            switch (inst.opcode())
            {
            case SwitchMachine:
                break;
            case Exit:
                return;
            case LdScratch: {
                int16_t* scratch = data->get_scratch(inst);
                SWITCH_REG([&] (reg_t r)
                {
                    memcpy(r, &scratch[myL1Offset], sizeof(reg_t));
                })
            }
            case StScratch: {
                int16_t* scratch = data->get_scratch(inst);
                SWITCH_REG([&] (reg_t r)
                {
                    memcpy(&scratch[myL1Offset], r, sizeof(reg_t));
                })
            }
            case AddFeature: {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t *weights = &transformer->weights[index * L1Size] + myL1Offset;
                    SWITCH_REG([&] (reg_t r)
                    {
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++) {
                            unsigned val;
                            memcpy(&val, &weights[2 * i], 4);
                            r[i] = __vadd2(r[i], val);
                        }
                    })
                } else {
                    const int8_t *weights = &transformer->threatWeights[index * L1Size] + myL1Offset;
                    SWITCH_REG(([&] (reg_t r)
                    {
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i += 2) {
                            unsigned val, l, h;
                            memcpy(&val, &weights[4 * i], 4);
                            cvt8_to_16(val, &l, &h);
                            r[i] = __vadd2(r[i], l);
                            r[i + 1] = __vadd2(r[i + 1], h);
                        }
                    }))
                }
                break;
            }
            case SubFeature:
            {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t *weights = &transformer->weights[index * L1Size] + myL1Offset;
                    SWITCH_REG([&] (reg_t r)
                    {
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++) {
                            unsigned val;
                            memcpy(&val, &weights[2 * i], 4);
                            r[i] = __vsub2(r[i], val);
                        }
                    })
                } else {
                    const int8_t *weights = &transformer->threatWeights[index * L1Size] + myL1Offset;
                    SWITCH_REG(([&] (reg_t r)
                    {
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i += 2) {
                            unsigned val, l, h;
                            memcpy(&val, &weights[4 * i], 4);
                            cvt8_to_16(val, &l, &h);
                            r[i] = __vsub2(r[i], l);
                            r[i + 1] = __vsub2(r[i + 1], h);
                        }
                    }))
                }
                break;
            }
            case ComputeL1: {
                /*Eval::NNUE::L1Bucket* bucket = &buckets[inst.decode_bucket()];
                int32_t data[16];

                for (int i = 0; i < PtxRegsPerThreadSlice; i++)
                {

                }

                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    for (int offset = 16; offset > 0; offset /= 2) {
                        data[i] += __shfl_down_sync(0xFFFFFFFF, data[i], offset);
                    }
                }*/
                break;
            }
            case ResetReg: {
                if (is_halfka_reg(inst.decode_reg()))
                {
                    SWITCH_REG([&] (reg_t r)
                    {
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++)
                        {
                            r[i] = 0;
                        }
                    })
                }

                break;
            }
            }

            if (lane_id == 0) {
                machine->tail = (machine->tail + 1) % InstructionQueueSize;
            }
        }
    }

    CudaContext::CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount): machineCount(machineCount), weights(std::make_unique<WeightsData>(big))
    {
        checkError(
            cudaHostAlloc(&machines, machineCount * sizeof(RegisterMachine), cudaHostAllocMapped)
        );

        memset(machines, 0, machineCount * sizeof(RegisterMachine));
        for (int i = 0; i < machineCount; i++) {
            RegisterMachine *machine = &machines[i];

            checkError(cudaMalloc(&machine->data, sizeof(RegisterData)));
        }
    }

    CudaContext::~CudaContext()
    {
        for (int i = 0; i < machineCount; i++)
        {
            cudaFreeHost(machines[i].data);
        }
        cudaFree(machines);
        machines = nullptr;
    }

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count)
    {
        return std::make_unique<CudaContext>(networks, machine_count);
    }
}
