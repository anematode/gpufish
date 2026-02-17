#include "gpu.h"
#include "gpu_defs.h"

#include <cstdio>
#include <memory>

#include <thread>
#include <x86gprintrin.h>

#include "engine.h"
#include "nnue/network.h"

// Credit: https://stackoverflow.com/a/14038590
#define checkError(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

static void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


namespace Stockfish::GPU {

constexpr int L1EntriesPerThreadSlice = L1Size / ThreadsPerWarp;
constexpr int PtxRegsPerThreadSlice =
  L1EntriesPerThreadSlice;  // each unsigned contains two 16-bit values

struct ScratchReg {
    int16_t data[L1Size];
};

// Device-side data that lives only in GPU memory
struct RegisterData {
    ScratchReg regs[ScratchRegCount];

    __device__ int16_t* get_scratch(Instruction inst) {
        return regs[inst.decode_wide_index()].data;
    }
};

struct WeightsData {
    // Device-side pointers
    Eval::NNUE::BigFeatureTransformer* transformer;
    Eval::NNUE::L1Bucket*              buckets;

    WeightsData(const Eval::NNUE::NetworkBig& big) {
        const Eval::NNUE::BigFeatureTransformer& transformer = big.get_feature_transformer();
        auto                                     sparse_input_buckets = big.get_sparse_buckets();

        auto temp = std::make_unique<Eval::NNUE::BigFeatureTransformer>(transformer);
        temp->unpermute_weights();

        checkError(cudaMalloc(&this->transformer, sizeof(*temp)));
        checkError(cudaMemcpy(this->transformer, &*temp, sizeof(*temp), cudaMemcpyHostToDevice));

        size_t bc = sparse_input_buckets.size();
        checkError(cudaMalloc(&buckets, sizeof(*sparse_input_buckets[0]) * bc));

        for (size_t i = 0; i < bc; i++)
        {
            auto biases  = sparse_input_buckets[i]->get_biases();
            auto weights = sparse_input_buckets[i]->get_weights();

            checkError(cudaMemcpy(&buckets[i].biases, biases.data(), sizeof(buckets[i].biases),
                                  cudaMemcpyHostToDevice));
            checkError(cudaMemcpy(&buckets[i].weights, weights.data(), sizeof(buckets[i].weights),
                                  cudaMemcpyHostToDevice));
        }
    }

    WeightsData(const WeightsData&) = delete;

    ~WeightsData() {
        checkError(cudaFree(transformer));
        transformer = nullptr;

        checkError(cudaFree(buckets));
        buckets = nullptr;
    }
};

// GPU-side, collectively managed by warps to perform work stealing. Important data members are also hoisted here
// to avoid expensive host accesses.
struct CachedMachineInfo
{
    int warp_id;
    uint32_t last_buffer_header;  // used to distinguish a new header

    RegisterData* regData;
    InstructionBuffer* wcBuffer;
};


__device__ static bool is_halfka_reg(Reg reg) { return reg == A || reg == B; }

enum ReduceOp {
    Add,
    Sub,
    Store
};

template<ReduceOp op>
__device__ void unpack16_to_32(int i, int& i1, int& i2) {
    switch (op)
    {
    case Add :
        i1 += i;  // we only care about the low 16 bits

        // dp2a.lo.b32 multiplies the 2 16-bit signed halfwords in first arg by the 2 low 8-bit signed bytes in the
        // second arg. Thus, this extracts the high 16 bits of the operand i and adds them to i2.
        i2 = __dp2a_lo(i, 1 << 8, i2);
        return;
    case Sub :
        i1 -= i;  // we only care about the low 16 bits
        i2 = __dp2a_lo(i, 0xff << 8, i2);
        return;
    case Store :
        i1 = i;
        i2 = i >> 16;
    }
}

template<ReduceOp op>
__device__ void unpack8_to_32(int i, int& i1, int& i2, int& i3, int& i4) {
    switch (op)
    {
    case Add :
        // see above comment -- analogous to __dp2a
        i1 = __dp4a(i, 0x1, i1);
        i2 = __dp4a(i, 0x1 << 8, i2);
        i3 = __dp4a(i, 0x1 << 16, i3);
        i4 = __dp4a(i, 0x1 << 24, i4);
        return;
    case Sub :
        i1 = __dp4a(i, 0xff, i1);
        i2 = __dp4a(i, 0xff << 8, i2);
        i3 = __dp4a(i, 0xff << 16, i3);
        i4 = __dp4a(i, 0xff << 24, i4);
        return;
    case Store :
        assert(false);
    }
}

__device__ int pack16(int i1, int i2, int& out) {
    // TODO: check this gets optimized to bfi.b32 or wtv
    out = (i1 & 0xffff) + (unsigned(i2) << 16);
}

__device__ void insert_byte(unsigned& i, int byte, int offset) {
    assert(offset < 4);

    int shamt = offset * 8;
    i |= byte << shamt;
}

constexpr int NO_WARP = -1;

__global__ void setup_machine_info(const RegisterMachine* machines, CachedMachineInfo* machine_infos,
                                   int num_machines)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_machines)
        return;

    const RegisterMachine& source = machines[idx];
    CachedMachineInfo& destination = machine_infos[idx];

    destination.warp_id = NO_WARP;
    destination.wcBuffer = source.wcBuffer;
    destination.last_buffer_header = 0;
    destination.regData = source.regData;
}

__device__ void
parallel_copy(Eval::NNUE::L1Bucket& dest, const Eval::NNUE::L1Bucket& src, unsigned lane_id) {
    using u32    = uint32_t __attribute__((may_alias));
    u32*       d = (u32*) &dest;
    const u32* s = (const u32*) &src;

    uint32_t count = sizeof(dest) / sizeof(u32);
    for (uint32_t i = lane_id; i < count; i += ThreadsPerWarp)
    {
        d[i] = s[i];
    }
}

__global__ void
persistent_kernel(RegisterMachine* machines, CachedMachineInfo* machine_infos, int num_machines) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerWarp;
    unsigned lane_id = threadIdx.x % ThreadsPerWarp;

    // Each warp picks a queue to monitor
    if (warp_id >= num_machines)
        return;

    // All are the same, so just use the first one
    auto*         transformer = machines[0].weights->transformer;
    auto*         buckets     = machines[0].weights->buckets;

    RegisterMachine*   machine           = &machines[warp_id];
    InstructionBuffer* instructionBuffer = machine_infos[warp_id].wcBuffer;
    RegisterData* regData        = machine_infos[warp_id].regData;

    typedef int reg_t[PtxRegsPerThreadSlice];
    reg_t       regA, regB, regC, regD;

    uint32_t           myL1Offset       = 8 * lane_id;
    constexpr uint32_t vectorLoadStride = 8 * ThreadsPerWarp;

    // To achieve good memory coalescing patterns, we implement the following indexing:
    // reg[i:i+7] = weights[myL1Offset+PtxRegsPerThreadSize/8*i:myL1Offset+PtxRegsPerThreadSize/8*i+7]

    uint32_t instructionCount = 0, signal = 0;

#define SWITCH_REG(X) \
    switch (inst.decode_reg()) \
    { \
    case 0 : { \
        auto& r = regA; \
        X; \
        break; \
    } \
    case 1 : { \
        auto& r = regB; \
        X; \
        break; \
    } \
    case 2 : { \
        auto& r = regC; \
        X; \
        break; \
    } \
    case 3 : { \
        auto& r = regD; \
        X; \
        break; \
    } \
    default : \
        __builtin_unreachable(); \
    };

#define SWITCH_REG_HALFKA(X) \
    switch (inst.decode_reg()) \
    { \
    case 0 : { \
        auto& r = regA; \
        X; \
        break; \
    } \
    case 1 : { \
        auto& r = regB; \
        X; \
        break; \
    } \
    default : \
        __builtin_unreachable(); \
    };

#define SWITCH_REG_THREATS(X) \
    switch (inst.decode_reg()) \
    { \
    case 2 : { \
        auto& r = regC; \
        X; \
        break; \
    } \
    case 3 : { \
        auto& r = regD; \
        X; \
        break; \
    } \
    default : \
        __builtin_unreachable(); \
    };

    constexpr int SharedMemoryBuckets = 4;

    __shared__ Eval::NNUE::L1Bucket bucketsShared[SharedMemoryBuckets];
    // bucketsShared[i - sharedBucketOffset], if in range, is buckets[i]
    int                             sharedBucketOffset = 8;

    __shared__ Instruction cmdBuffers[MaxInstructionsCount * 4];
    Instruction*           myCmdBuffer = &cmdBuffers[warp_id % 4];

    while (true)
    {
        // Warp leader polls the queue
        if (lane_id == 0)
        {
            uint32_t temp;
            while ((temp = *(volatile uint32_t*) &instructionBuffer->header) == signal)
            {
                __nanosleep(50);  // TODO better approach here?
            }
            signal           = temp;
            instructionCount = signal & 0xffff;
        }

        __syncwarp();
        instructionCount = __shfl_sync(0xFFFFFFFF, instructionCount, 0);

        if (instructionCount == MachineStopHeader)
        {
            machine->result[0] = 0;
            return;
        }

        // Copy instructions into shared memory
        for (uint32_t i = lane_id; i < instructionCount; i += ThreadsPerWarp)
        {
            myCmdBuffer[i] = instructionBuffer->list[i];
        }

        for (uint32_t inst_i = 0; inst_i < instructionCount; ++inst_i)
        {
            __syncwarp();

            Instruction inst = myCmdBuffer[inst_i];
            switch (inst.opcode())
            {
            case SwitchMachine :
                break;
            case PreloadL1Buckets : {
                sharedBucketOffset = inst.decode_bucket() - SharedMemoryBuckets + 1;

                for (int i = 0; i < SharedMemoryBuckets; ++i)
                {
                    int source = i + sharedBucketOffset;
                    if (source < 0 || source >= 8)
                        continue;

                    parallel_copy(bucketsShared[i], buckets[source], lane_id);
                }
                break;
            }
            case LdScratch : {
                int16_t* scratch = regData->get_scratch(inst);
                SWITCH_REG({
                    _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                           i += vectorLoadStride, j += 8) {
                        int4 data = *(int4*) &scratch[i];
                        unpack16_to_32<Store>(data.x, r[j], r[j + 1]);
                        unpack16_to_32<Store>(data.y, r[j + 2], r[j + 3]);
                        unpack16_to_32<Store>(data.z, r[j + 4], r[j + 5]);
                        unpack16_to_32<Store>(data.w, r[j + 6], r[j + 7]);
                    }
                })
                break;
            }
            case StScratch : {
                int16_t* scratch = regData->get_scratch(inst);
                SWITCH_REG({
                    _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                           i += vectorLoadStride, j += 8) {
                        int4 result;
                        pack16(r[j], r[j + 1], result.x);
                        pack16(r[j + 2], r[j + 3], result.y);
                        pack16(r[j + 4], r[j + 5], result.z);
                        pack16(r[j + 6], r[j + 7], result.w);
                        *(int4*) &scratch[i] = result;
                    }
                })

                break;
            }
            case AddFeature : {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t* weights = &transformer->weights[index * L1Size];
                    SWITCH_REG_HALFKA({
                        _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                               i += vectorLoadStride, j += 8) {
                            int4 data = *(int4*) &weights[i];
                            unpack16_to_32<Add>(data.x, r[j], r[j + 1]);
                            unpack16_to_32<Add>(data.y, r[j + 2], r[j + 3]);
                            unpack16_to_32<Add>(data.z, r[j + 4], r[j + 5]);
                            unpack16_to_32<Add>(data.w, r[j + 6], r[j + 7]);
                        }
                    })
                }
                else
                {
                    const int8_t* weights = &transformer->threatWeights[index * L1Size];
                    SWITCH_REG_THREATS(({
                        _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                               i += vectorLoadStride, j += 8) {
                            int2 data = *(int2*) &weights[i];
                            unpack8_to_32<Add>(data.x, r[j], r[j + 1], r[j + 2], r[j + 3]);
                            unpack8_to_32<Add>(data.y, r[j + 4], r[j + 5], r[j + 6], r[j + 7]);
                        }
                    }))
                }
                break;
            }
            case SubFeature : {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t* weights = &transformer->weights[index * L1Size];
                    SWITCH_REG_HALFKA({
                        _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                               i += vectorLoadStride, j += 8) {
                            int4 data = *(int4*) &weights[i];
                            unpack16_to_32<Sub>(data.x, r[j], r[j + 1]);
                            unpack16_to_32<Sub>(data.y, r[j + 2], r[j + 3]);
                            unpack16_to_32<Sub>(data.z, r[j + 4], r[j + 5]);
                            unpack16_to_32<Sub>(data.w, r[j + 6], r[j + 7]);
                        }
                    })
                }
                else
                {
                    const int8_t* weights = &transformer->threatWeights[index * L1Size];
                    SWITCH_REG_THREATS(({
                        _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                               i += vectorLoadStride, j += 8) {
                            int2 data = *(int2*) &weights[i];
                            unpack8_to_32<Sub>(data.x, r[j], r[j + 1], r[j + 2], r[j + 3]);
                            unpack8_to_32<Sub>(data.y, r[j + 4], r[j + 5], r[j + 6], r[j + 7]);
                        }
                    }))
                }
                break;
            }
            case Finalize : {
                // Pairwise multiplication values
                unsigned packed[L1EntriesPerThreadSlice / 4] = {0};

                auto to16 = [](int a) { return a << 16 >> 16; };

#pragma unroll
                for (int p = 0; p < 2; ++p)
                {
                    int32_t* src1   = p ? regB : regA;
                    int32_t* src2   = p ? regD : regC;
                    int      offset = p * (L1EntriesPerThreadSlice / 8);
#pragma unroll
                    for (int i = 0; i < L1EntriesPerThreadSlice / 2; ++i)
                    {
                        int sum0 = std::clamp(to16(src1[i] + src2[i]), 0, 255);
                        int sum1 = std::clamp(to16(src1[i + L1EntriesPerThreadSlice / 2]
                                                   + src2[i + L1EntriesPerThreadSlice / 2]),
                                              0, 255);

                        insert_byte(packed[offset + i / 4], unsigned(sum0 * sum1) / 512, i % 4);
                    }
                }

                // If it's black to move, we need to swap perspectives; because of our register layout
                // this is equivalent to exchanging the low and high halves within a thread
                if (inst.side_to_move())
                {
#pragma unroll
                    for (int i = 0; i < L1EntriesPerThreadSlice / 8; ++i)
                    {
                        int tmp = packed[i + L1EntriesPerThreadSlice / 8];
                        packed[i + L1EntriesPerThreadSlice / 8] = packed[i];
                        packed[i]                               = tmp;
                    }
                }

                int                   bucketIndex = inst.decode_bucket();
                Eval::NNUE::L1Bucket* bucket;

                unsigned sharedIndex = bucketIndex - sharedBucketOffset;
                if (sharedIndex < SharedMemoryBuckets)
                    bucket = &bucketsShared[sharedIndex];
                else
                    bucket = &buckets[bucketIndex];

                int accumulation = lane_id < 16 ? bucket->biases[lane_id] : 0;
                
#pragma unroll
                for (int j = 0, q = lane_id >= 16; j < L1EntriesPerThreadSlice / 4; j += 2)
                {
                    unsigned nnz = __ballot_sync(0xFFFFFFFF, packed[j] | packed[j + 1]);
                    while (nnz)
                    {
                        int th_i = __ffs(nnz) - 1;
                        nnz &= nnz - 1;

                        int b1 = __shfl_sync(0xFFFFFFFF, packed[j], th_i);
                        int b2 = __shfl_sync(0xFFFFFFFF, packed[j + 1], th_i);

                        int selected = lane_id < 16 ? b1 : b2;
                        accumulation =
                          __dp4a(selected,
                                 *((int*) &bucket->weights[64 * (q + 2 * th_i)] + (lane_id % 16)),
                                 accumulation);
                    }

                    q += ThreadsPerWarp * 2;
                }

                accumulation += __shfl_down_sync(0xFFFFFFFF, accumulation, 16);

                if (lane_id < 16)
                    machine->result[lane_id] = accumulation;
                __threadfence_system();  // tbh I don't understand why this is necessary but *shrug*
                goto done;
            }
            case ResetReg : {
                // Not performance critical, used just for resetting
                if (is_halfka_reg(inst.decode_reg()))
                {
                    SWITCH_REG_HALFKA({
                        _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                               i += vectorLoadStride, j += 8) {
                            int4 data = *(int4*) &transformer->biases.data()[i];
                            unpack16_to_32<Store>(data.x, r[j + 0], r[j + 1]);
                            unpack16_to_32<Store>(data.y, r[j + 2], r[j + 3]);
                            unpack16_to_32<Store>(data.z, r[j + 4], r[j + 5]);
                            unpack16_to_32<Store>(data.w, r[j + 6], r[j + 7]);
                        }
                    })
                }
                else
                {
                    SWITCH_REG_THREATS({
                        _Pragma("unroll") for (int i = 0; i < PtxRegsPerThreadSlice; i++) r[i] = 0;
                    })
                }

                break;
            }
            }
        }

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

RegisterMachine::RegisterMachine(const WeightsData* weights,
                                 InstructionBuffer* wcInstructionBuffer) {
    cudaStreamCreate((cudaStream_t*) &stream);
    checkError(cudaMalloc(&regData, sizeof(RegisterData)));
    checkError(cudaMemset(regData, 0, sizeof(RegisterData)));

    this->weights  = weights;
    this->wcBuffer = wcInstructionBuffer;
}

RegisterMachine::~RegisterMachine() {
    cudaStreamDestroy((cudaStream_t) stream);
    cudaFree(regData);
    regData     = nullptr;
    wcBuffer = nullptr;
    stream   = nullptr;
}

std::array<int16_t, 1024> RegisterMachine::read_scratch(size_t index) const {
    // We're not synchronizing with the persistent kernel's stream, so this is shoddy, but this is for debug
    // purposes so we just do this.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    std::array<int16_t, 1024> array;
    std::fill(array.begin(), array.end(), 5);
    cudaStream_t stream;
    checkError(cudaStreamCreate(&stream));
    checkError(
      cudaMemcpyAsync(&array, &regData->regs[index], sizeof(array), cudaMemcpyDeviceToHost, stream));
    checkError(cudaStreamSynchronize(stream));
    checkError(cudaStreamDestroy(stream));
    return array;
}

CudaContext::CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount) :
    machineCount(machineCount),
    weights(std::make_unique<WeightsData>(big)) {
    checkError(
      cudaHostAlloc(&machines, machineCount * sizeof(RegisterMachine), cudaHostAllocMapped));
    checkError(cudaHostAlloc(&wcBuffers, machineCount * sizeof(InstructionBuffer),
                             cudaHostAllocMapped | cudaHostAllocWriteCombined));
    checkError(cudaMalloc(&machineInfos, sizeof(CachedMachineInfo) * machineCount));

    memset(wcBuffers, 0, machineCount * sizeof(InstructionBuffer));

    for (int i = 0; i < machineCount; i++)
    {
        new (&machines[i]) RegisterMachine(weights.get(), &wcBuffers[i]);
    }
}

void CudaContext::stop_all() {
    if (!stream)
        return;

    // Stop all machines
    for (size_t i = 0; i < machineCount; i++)
    {
        machines[i].setStopSignal();
        machines[i].blockUntilComplete();
        machines[i].isActive = false;
    }

    cudaStreamSynchronize((cudaStream_t) stream);
    cudaStreamDestroy((cudaStream_t) stream);
    stream = nullptr;
}

RegisterMachine* CudaContext::get_machine(size_t size) {
    assert(size < machineCount);
    return &machines[size];
}

CudaContext::~CudaContext() {
    stop_all();

    for (int i = 0; i < machineCount; i++)
    {
        machines[i].~RegisterMachine();
    }
    cudaFreeHost(wcBuffers);
    cudaFree(machineInfos);
    cudaFree(machines);
    machines  = nullptr;
    wcBuffers = nullptr;
    machineInfos = nullptr;
}

void CudaContext::launch_persistent_kernel() {
    if (stream)
        return;

    checkError(cudaStreamCreate((cudaStream_t*) &stream));
    memset(wcBuffers, 0, machineCount * sizeof(InstructionBuffer));

    int num_warps         = machineCount;
    int threads_per_block = 128;
    int num_blocks        = (num_warps * 32 + threads_per_block - 1) / threads_per_block;

    for (size_t i = 0; i < machineCount; i++)
    {
        machines[i].isActive = true;
    }

    setup_machine_info<<<machineCount / 32 + 1, 32, 0>>>(machines, machineInfos, machineCount);

    persistent_kernel<<<num_blocks, threads_per_block, 0, (cudaStream_t) stream>>>(
      machines, machineInfos, machineCount);
}
}
