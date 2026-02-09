#include "gpu.h"
#include "gpu_defs.h"

#include <cstdio>
#include <memory>

#include "nnue/network.h"

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

        __device__ int16_t *get_scratch(Instruction inst)
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
            checkError(cudaMemcpy(this->transformer, &*temp, sizeof(transformer), cudaMemcpyHostToDevice));

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

    __device__ void cvt8_to_16(uint32_t data, uint32_t *l, uint32_t *h)
    {
        uint32_t lo, hi;
        asm ("prmt.b32 %0,%2,0,0x9180;\n"
             "prmt.b32 %1,%2,0,0xb3a2;" :  "=r"(lo), "=r"(hi) : "r"(data));
        *l = lo;
        *h = hi;
    }

    __device__ bool is_halfka_reg(Reg reg)
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

        Instruction* instructionBuffer = machine->queue;
        auto* instructionCountPtr = &machine->instructionCount;

        typedef unsigned reg_t[PtxRegsPerThreadSlice];
        reg_t regA, regB, regC, regD;

        uint32_t myL1Offset = L1EntriesPerThreadSlice * lane_id;
        uint32_t instructionCount = 0;

#define SWITCH_REG(X) switch (inst.decode_reg()) { \
    case 0: { X(regA); break; } \
    case 1: { X(regB); break; } \
    case 2: { X(regC); break; } \
    case 3: { X(regD); break; } \
    default: __builtin_unreachable(); \
        };

        __shared__ Instruction cmdBuffers[MaxInstructionsCount * 4];
        Instruction* myCmdBuffer = &cmdBuffers[warp_id % 4];

        while (true) {
            // Warp leader polls the queue
            if (lane_id == 0) {
                while ((instructionCount = *instructionCountPtr) == 0)
                {
                    __nanosleep(50);  // TODO better approach here?
                }
            }

            uint32_t mask = __activemask();
            instructionCount = __shfl_sync(0xFFFFFFFF, instructionCount, 0);
            // Copy instructions into shared memory
            for (uint32_t i = lane_id; i < instructionCount; i += ThreadsPerWarp)
            {
                myCmdBuffer[i] = instructionBuffer[i];
            }

            for (uint32_t inst_i = 0; inst_i < instructionCount; ++inst_i)
            {
                const Instruction& inst = myCmdBuffer[inst_i];
                switch (inst.opcode())
                {
                case SwitchMachine:
                    break;
                case Exit:
                    {
                        machine->result[0] = 0;
                        return;
                    }
                case LdScratch: {
                        int16_t* scratch = data->get_scratch(inst);
                        SWITCH_REG([&] (reg_t r)
                        {
                            memcpy(r, &scratch[myL1Offset], sizeof(reg_t));
                        })
                        break;
                }
                case StScratch: {
                        int16_t* scratch = data->get_scratch(inst);
                        SWITCH_REG([&] (reg_t r)
                        {
                            memcpy(&scratch[myL1Offset], r, sizeof(reg_t));
                        })
                        break;
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
                case Finalize: {
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

                        machine->result[0] = 1;
                        break;
                }
                case ResetReg: {
                        if (is_halfka_reg(inst.decode_reg()))
                        {
                            SWITCH_REG([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++)
                                {
                                    uint32_t val;
                                    memcpy(&val, transformer->biases.data() + myL1Offset + 2 * i, 4);
                                    r[i] = val;
                                }
                            })
                        } else
                        {
                            SWITCH_REG([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++)
                                    r[i] = 0;
                            })
                        }

                        break;
                }
                }
            }

            // Signal to the CPU that we're done with this batch
            if (lane_id == 0)
            {
                *instructionCountPtr = 0;
                machine->result[0] = 1;
            }
        }
    }

    void RegisterMachine::init()
    {
        cudaStreamCreate((cudaStream_t*) &stream);
        checkError(cudaMalloc(&data, sizeof(RegisterData)));
        checkError(cudaMemset(data, 0, sizeof(RegisterData)));
    }

    void RegisterMachine::deinit()
    {
        cudaStreamDestroy((cudaStream_t) stream);
        cudaFree(data);
        stream = nullptr;
    }

    void RegisterMachine::submit(Instruction instr)
    {
        if (!isActive)
        {
            fprintf(stderr, "RegisterMachine is inactive!\n");
            abort();
        }
        if (queueIndex >= MaxInstructionsCount)
        {
            // Need an immediate flush before writing the next instruction
            // Mainly used during setup
            flush();
            blockUntilComplete();
        }
        queue[queueIndex++] = instr;
    }

    void RegisterMachine::flush()
    {
        if (queueIndex == 0)
            return;
        std::fill_n(result, 16, INT_MIN);
        instructionCount = queueIndex;
    }

    void RegisterMachine::blockUntilComplete()
    {
        while (!ready())  // TODO add a "perf counter" for this
        {
            asm("pause");
        }

        // TODO verify that all entries are written

        queueIndex = 0;
        instructionCount = 0;
    }

    bool RegisterMachine::ready() const
    {
        return result[0] != INT_MIN || instructionCount == 0;
    }

    std::array<int16_t, 1024> RegisterMachine::read_scratch(size_t index)
    {
        std::array<int16_t, 1024> array;
        checkError(cudaMemcpy(&array, &data->regs[index], sizeof(array), cudaMemcpyDeviceToHost));
        return array;
    }

    CudaContext::CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount): machineCount(machineCount), weights(std::make_unique<WeightsData>(big))
    {
        checkError(
            cudaHostAlloc(&machines, machineCount * sizeof(RegisterMachine), cudaHostAllocMapped)
        );

        memset(machines, 0, machineCount * sizeof(RegisterMachine));
        for (int i = 0; i < machineCount; i++) {
            RegisterMachine *machine = &machines[i];

            machine->init();
            machine->weights = weights.get();
        }
    }

    void CudaContext::stop_all()
    {
        if (!stream)
            return;

        // Stop all machines
        for (size_t i = 0; i < machineCount; i++)
        {
            machines[i].submit(Instruction::stop());
            machines[i].flush();
            machines[i].blockUntilComplete();
            machines[i].isActive = false;
        }

        cudaStreamSynchronize((cudaStream_t) stream);
        cudaStreamDestroy((cudaStream_t) stream);
        stream = nullptr;
    }

    RegisterMachine* CudaContext::get_machine(size_t size)
    {
        assert(size < machineCount);
        return &machines[size];
    }

    CudaContext::~CudaContext()
    {
        stop_all();

        for (int i = 0; i < machineCount; i++)
        {
            machines[i].deinit();
            cudaFreeHost(machines[i].data);
        }
        cudaFree(machines);
        machines = nullptr;
    }

    void CudaContext::launch_persistent_kernel()
    {
        if (stream)
            return;

        checkError(cudaStreamCreate((cudaStream_t*) &stream));

        int num_warps = machineCount;
        int threads_per_block = 128;
        int num_blocks = (num_warps * 32 + threads_per_block - 1) / threads_per_block;

        for (size_t i = 0; i < machineCount; i++)
        {
            machines[i].isActive = true;
        }

        persistent_kernel<<<num_blocks, threads_per_block, 0, (cudaStream_t) stream>>>(machines, machineCount);
    }

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count)
    {
        return std::make_unique<CudaContext>(networks, machine_count);
    }
}
