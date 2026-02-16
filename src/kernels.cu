#include "gpu.h"
#include "gpu_defs.h"

#include <cstdio>
#include <memory>

#include <thread>
#include <x86gprintrin.h>
#include <clflushoptintrin.h>
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
    constexpr int PtxRegsPerThreadSlice = L1EntriesPerThreadSlice;  // each unsigned contains two 16-bit values

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
            std::cout << "Constructing a WeightsData!\n";
            const Eval::NNUE::BigFeatureTransformer& transformer = big.get_ft();
            auto sparse_input_buckets = big.get_input_buckets();

            auto temp = std::make_unique<Eval::NNUE::BigFeatureTransformer>(transformer);
            temp->unpermute_weights();

            checkError(cudaMalloc(&this->transformer, sizeof(*temp)));
            checkError(cudaMemcpy(this->transformer, &*temp, sizeof(*temp), cudaMemcpyHostToDevice));

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


    __device__ bool is_halfka_reg(Reg reg)
    {
        return reg == A || reg == B;
    }

    enum ReduceOp
    {
        Add, Sub, Store
    };

    __device__ void unpack16_to_32(int i, int& i1, int& i2, ReduceOp op)
    {
        switch (op)
        {
        case Add:
            i1 += i; // we only care about the low 16 bits
            i2 += (i >> 16);
            return;
        case Sub:
            i1 -= i; // we only care about the low 16 bits
            i2 -= (i >> 16);
            return;
        case Store:
            i1 = i;
            i2 = i >> 16;
        }
    }

    __device__ void unpack8_to_32(int i, int& i1, int& i2, int& i3, int& i4, ReduceOp op)
    {
        switch (op)
        {
        case Add:
            i1 = __dp4a(i, 0x1, i1);
            i2 = __dp4a(i, 0x1 << 8, i2);
            i3 = __dp4a(i, 0x1 << 16, i3);
            i4 = __dp4a(i, 0x1 << 24, i4);
            return;
        case Sub:
            i1 = __dp4a(i, 0xff, i1);
            i2 = __dp4a(i, 0xff << 8, i2);
            i3 = __dp4a(i, 0xff << 16, i3);
            i4 = __dp4a(i, 0xff << 24, i4);
            return;
        case Store:
            assert(false);
        }
    }

    __device__ int pack16(int i1, int i2, int& out)
    {
        out = (i1 & 0xffff) + (unsigned(i2) << 16);
    }

    __device__ void insert_byte(unsigned& i, int byte, int offset)
    {
        assert(offset < 4);

        int shamt = offset * 8;
        i |= byte << shamt;
    }

    __global__ void persistent_kernel(RegisterMachine* machines, WCInstructionBuffer* buffers, int num_machines) {
        unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerWarp;
        unsigned lane_id = threadIdx.x % ThreadsPerWarp;

        // Each warp picks a queue to monitor
        if (warp_id >= num_machines) return;

        RegisterMachine *machine = &machines[warp_id];
        WCInstructionBuffer* instructionBuffer = &buffers[warp_id];

        RegisterData *data = machine->data;
        auto* transformer = machine->weights->transformer;
        auto* buckets = machine->weights->buckets;

        typedef int reg_t[PtxRegsPerThreadSlice];
        reg_t regA, regB, regC, regD;

        uint32_t myL1Offset = 8 * lane_id;
        constexpr uint32_t vectorLoadStride = 8 * ThreadsPerWarp;

        // To achieve good memory coalescing patterns, we implement the following indexing:
        // reg[i:i+7] = weights[myL1Offset+PtxRegsPerThreadSize/8*i:myL1Offset+PtxRegsPerThreadSize/8*i+7]

        uint32_t instructionCount = 0, signal = 0;

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
                uint32_t temp;
                while ((temp = *(volatile uint32_t*)&instructionBuffer->data) == signal)
                {
                    __nanosleep(50);  // TODO better approach here?
                }
                signal = temp;
                instructionCount = signal & 0xffff;
            }

            __syncwarp();
            instructionCount = __shfl_sync(0xFFFFFFFF, instructionCount, 0);

            // Copy instructions into shared memory
            for (uint32_t i = lane_id; i < instructionCount; i += ThreadsPerWarp)
            {
                myCmdBuffer[i] = instructionBuffer->list[i];
            }

            for (uint32_t inst_i = 0; inst_i < instructionCount; ++inst_i)
            {
                __syncwarp();

                const Instruction& inst = myCmdBuffer[inst_i];
                switch (inst.opcode())
                {
                case SwitchMachine:
                    break;
                case Exit:
                    {
                        if (lane_id == 0)
                        {
                            machine->result[0] = 0;
                            __threadfence_system();
                        }
                        return;
                    }
                case LdScratch: {
                        int16_t* scratch = data->get_scratch(inst);
                        SWITCH_REG([&] (reg_t r)
                        {
                            _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8)
                            {
                                int4 data = *(int4*)&scratch[i];
                                unpack16_to_32(data.x, r[j], r[j + 1], Store);
                                unpack16_to_32(data.y, r[j + 2], r[j + 3], Store);
                                unpack16_to_32(data.z, r[j + 4], r[j + 5], Store);
                                unpack16_to_32(data.w, r[j + 6], r[j + 7], Store);
                            }
                        })
                        break;
                }
                case StScratch: {
                        int16_t* scratch = data->get_scratch(inst);
                        SWITCH_REG([&] (reg_t r)
                        {
                            _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8)
                            {
                                int4 result;
                                pack16(r[j], r[j+1], result.x);
                                pack16(r[j+2], r[j+3], result.y);
                                pack16(r[j+4], r[j+5], result.z);
                                pack16(r[j+6], r[j+7], result.w);
                                *(int4*)&scratch[i] = result;
                            }
                        })

                        break;
                }
                case AddFeature: {
                        uint32_t index = inst.decode_wide_index();
                        if (is_halfka_reg(inst.decode_reg()))
                        {
                            const int16_t *weights = &transformer->weights[index * L1Size];
                            SWITCH_REG([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8) {
                                    int4 data = *(int4*)&weights[i];
                                    unpack16_to_32(data.x, r[j], r[j + 1], Add);
                                    unpack16_to_32(data.y, r[j + 2], r[j + 3], Add);
                                    unpack16_to_32(data.z, r[j + 4], r[j + 5], Add);
                                    unpack16_to_32(data.w, r[j + 6], r[j + 7], Add);
                                }
                            })
                        } else {
                            const int8_t *weights = &transformer->threatWeights[index * L1Size];
                            SWITCH_REG(([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8) {
                                    int2 data = *(int2*)&weights[i];
                                    unpack8_to_32(data.x, r[j], r[j + 1], r[j + 2], r[j + 3], Add);
                                    unpack8_to_32(data.y, r[j + 4], r[j + 5], r[j + 6], r[j + 7], Add);
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
                            const int16_t *weights = &transformer->weights[index * L1Size];
                            SWITCH_REG([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8) {
                                    int4 data = *(int4*)&weights[i];
                                    unpack16_to_32(data.x, r[j], r[j + 1], Sub);
                                    unpack16_to_32(data.y, r[j + 2], r[j + 3], Sub);
                                    unpack16_to_32(data.z, r[j + 4], r[j + 5], Sub);
                                    unpack16_to_32(data.w, r[j + 6], r[j + 7], Sub);
                                }
                            })
                        } else {
                            const int8_t *weights = &transformer->threatWeights[index * L1Size];
                            SWITCH_REG(([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8) {
                                    int2 data = *(int2*)&weights[i];
                                    unpack8_to_32(data.x, r[j], r[j + 1], r[j + 2], r[j + 3], Sub);
                                    unpack8_to_32(data.y, r[j + 4], r[j + 5], r[j + 6], r[j + 7], Sub);
                                }
                            }))
                        }
                        break;
                    }
                case Finalize: {
                        // First compute pairwise multiplication values
                        unsigned packed[L1EntriesPerThreadSlice / 4] = {0};

                        auto cvt = [] (int a)
                        {
                            return a << 16 >> 16;
                        };

#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            int32_t *src1 = p ? regB : regA;
                            int32_t *src2 = p ? regD : regC;
                            int offset = p * (L1EntriesPerThreadSlice / 8);
#pragma unroll
                            for (int i = 0; i < L1EntriesPerThreadSlice / 2; ++i) {
                                int sum0 = std::clamp(cvt(src1[i] + src2[i]), 0, 255);
                                int sum1 = std::clamp(cvt(src1[i+ L1EntriesPerThreadSlice / 2] + src2[i + L1EntriesPerThreadSlice / 2]), 0, 255);

                                insert_byte(packed[offset + i / 4], unsigned(__mul24(sum0, sum1)) / 512, i % 4);
                            }
                        }

                        if (inst.side_to_move())
                        {
#pragma unroll
                            for (int i = 0; i < L1EntriesPerThreadSlice / 8; ++i)
                            {
                                int tmp = packed[i + L1EntriesPerThreadSlice / 8];
                                packed[i + L1EntriesPerThreadSlice / 8] = packed[i];
                                packed[i] = tmp;
                            }
                        }

                        Eval::NNUE::L1Bucket* bucket = &buckets[inst.decode_bucket()];
                        int accumulation = lane_id < 16 ? bucket->biases[lane_id] : 0;

                        __syncwarp();
#pragma unroll
                        for (int j = 0, q = lane_id >= 16; j < L1EntriesPerThreadSlice / 4; j += 2)
                        {
                            for (int th_i = 0; th_i < ThreadsPerWarp; ++th_i, q += 2)
                            {
                                int b1 = __shfl_sync(0xFFFFFFFF, packed[j], th_i);
                                int b2 = __shfl_sync(0xFFFFFFFF, packed[j + 1], th_i);

                                int selected = lane_id < 16 ? b1 : b2;
                                if (selected)
                                {
                                    accumulation = __dp4a(selected, *((int*)&bucket->weights[64 * q] + (lane_id % 16)), accumulation);
                                }
                            }
                        }

                        accumulation += __shfl_down_sync(0xFFFFFFFF, accumulation, 16);

                        if (lane_id < 16)
                            machine->result[lane_id] = accumulation;
                        __threadfence_system(); // tbh I don't understand why this is necessary but *shrug*
                        goto done;
                }
                case ResetReg: {
                        // Not performance critical, used just for resetting
                        if (is_halfka_reg(inst.decode_reg()))
                        {
                            SWITCH_REG([&] (reg_t r)
                            {
                                _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 8)
                                {
                                    int4 data = *(int4*)&transformer->biases.data()[i];
                                    unpack16_to_32(data.x, r[j + 0], r[j + 1], Store);
                                    unpack16_to_32(data.y, r[j + 2], r[j + 3], Store);
                                    unpack16_to_32(data.z, r[j + 4], r[j + 5], Store);
                                    unpack16_to_32(data.w, r[j + 6], r[j + 7], Store);
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

            __syncwarp();
            // Signal to the CPU that we're done with this batch
            if (lane_id == 0)
            {
                machine->result[0] = 1;
                __threadfence_system();
            }

            done:;
            __syncwarp();
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
        data = nullptr;
        wcBuffer = nullptr;
        stream = nullptr;
    }

    void RegisterMachine::submit(Instruction instr)
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
            if (regStates[instr.decode_reg()] == instr.decode_wide_index())
            {
                return;
            }
            break;
        case StScratch:
            regStates[instr.decode_reg()] = instr.decode_wide_index();
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

    void RegisterMachine::flush()
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

    void RegisterMachine::blockUntilComplete()
    {
        int attempts = 0;
        while (!ready())  // TODO add a "perf counter" for this
        {
            // asm("clflush %0" :: "m"(result[0]));
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

    bool RegisterMachine::ready() const
    {
        return result[0] != INT_MIN;
    }

    std::array<int16_t, 1024> RegisterMachine::read_scratch(size_t index)
    {
        // We're not synchronizing with the persistent kernel's stream, so this is shoddy, but this is for debug
        // purposes so we just do this.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

        std::array<int16_t, 1024> array;
        std::fill(array.begin(), array.end(), 5);
        cudaStream_t stream;
        checkError(cudaStreamCreate(&stream));
        checkError(cudaMemcpyAsync(&array, &data->regs[index], sizeof(array), cudaMemcpyDeviceToHost, stream));
        checkError(cudaStreamSynchronize(stream));
        checkError(cudaStreamDestroy(stream));
        return array;
    }

    CudaContext::CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount): machineCount(machineCount), weights(std::make_unique<WeightsData>(big))
    {
        checkError(
            cudaHostAlloc(&machines, machineCount * sizeof(RegisterMachine), cudaHostAllocMapped)
        );
        checkError(
            cudaHostAlloc(&wcBuffers, machineCount * sizeof(WCInstructionBuffer), cudaHostAllocMapped | cudaHostAllocWriteCombined)
        );

        memset(machines, 0, machineCount * sizeof(RegisterMachine));
        memset(wcBuffers, 0, machineCount * sizeof(WCInstructionBuffer));
        for (int i = 0; i < machineCount; i++) {
            RegisterMachine *machine = &machines[i];

            machine->init();
            machine->weights = weights.get();
            machine->wcBuffer = &wcBuffers[i];
        }
    }

    void CudaContext::stop_all()
    {
        if (!stream)
            return;

        // Stop all machines
        for (size_t i = 0; i < machineCount; i++)
        {
            machines[i].blockUntilComplete();
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
        cudaFreeHost(wcBuffers);
        cudaFree(machines);
        machines = nullptr;
        wcBuffers = nullptr;
    }

    void CudaContext::launch_persistent_kernel()
    {
        if (stream)
            return;

        checkError(cudaStreamCreate((cudaStream_t*) &stream));
        memset(wcBuffers, 0, machineCount * sizeof(WCInstructionBuffer));

        int num_warps = machineCount;
        int threads_per_block = 128;
        int num_blocks = (num_warps * 32 + threads_per_block - 1) / threads_per_block;

        for (size_t i = 0; i < machineCount; i++)
        {
            machines[i].isActive = true;
        }

        persistent_kernel<<<num_blocks, threads_per_block, 0, (cudaStream_t) stream>>>(machines, wcBuffers, machineCount);
    }

    std::unique_ptr<CudaContext> make_context(const Eval::NNUE::NetworkBig& networks, size_t machine_count)
    {
        return std::make_unique<CudaContext>(networks, machine_count);
    }
}
