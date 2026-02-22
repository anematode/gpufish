#include "gpu.h"
#include "gpu_defs.h"

#include <cstdio>
#include <memory>
#include <cuda_pipeline.h>

#include <thread>
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

constexpr int WarpsPerThreadBlock     = 16;
constexpr int L1EntriesPerThreadSlice = L1Size / ThreadsPerWarp;
// each unsigned contains two 16-bit values
constexpr int PtxRegsPerThreadSlice = L1EntriesPerThreadSlice / 2;

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

        // We need to mask out the high bit of each 16-bit halfKA weight/bias because this makes SWAR more efficient
        // in the kernel
        for (auto& w : temp->weights)
            w &= 0x7fff;
        for (auto& w : temp->biases)
            w &= 0x7fff;

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


__device__ static bool is_halfka_reg(Reg reg) { return reg == A || reg == B; }

enum ReduceOp {
    Add,
    Sub,
    Store
};

template<ReduceOp op>
__device__ void unpack16_to_32(unsigned i, unsigned& i1) {
    assert((i & 0x80008000) == 0);
    assert((i1 & 0x80008000) == 0);

    switch (op)
    {
    case Add :
        i1 += i;
        i1 &= 0x7fff7fff;
        return;
    case Sub :
        i1 |= 0x80008000U;
        i1 -= i;
        i1 &= 0x7fff7fff;
        return;
    case Store :
        i1 = i;
    }
}

__device__ void cvt8_to_16(uint32_t data, uint32_t* l, uint32_t* h) {
    uint32_t lo, hi;
    asm("prmt.b32 %0,%2,0,0x9180;\n"
        "prmt.b32 %1,%2,0,0xb3a2;"
        : "=r"(lo), "=r"(hi)
        : "r"(data));
    *l = lo;
    *h = hi;
}

template<ReduceOp op>
__device__ void unpack8_to_32(int i, unsigned& i1, unsigned& i2) {
    uint32_t lo, hi;
    cvt8_to_16(i, &lo, &hi);
    lo &= 0x7fff7fff;
    hi &= 0x7fff7fff;

    unpack16_to_32<op>(lo, i1);
    unpack16_to_32<op>(hi, i2);
}

__device__ void insert_byte(unsigned& i, int byte, int offset) {
    assert(offset < 4);

    int shamt = offset * 8;
    i |= byte << shamt;
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
persistent_kernel(RegisterMachine* machines, InstructionBuffer* buffers, int num_machines) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerWarp;
    unsigned lane_id = threadIdx.x % ThreadsPerWarp;

    // Each warp picks a queue to monitor
    if (warp_id >= num_machines)
        return;

    RegisterMachine*   machine           = &machines[warp_id];
    InstructionBuffer* instructionBuffer = &buffers[warp_id];

    RegisterData* data        = machine->data;
    auto*         transformer = machine->weights->transformer;
    auto*         buckets     = machine->weights->buckets;

    typedef unsigned reg_t[PtxRegsPerThreadSlice];
    reg_t            regA, regC;

    // Pairwise multiplication values
    unsigned packed[L1EntriesPerThreadSlice / 4] = {0};

    uint32_t           myL1Offset       = 8 * lane_id;
    constexpr uint32_t vectorLoadStride = 8 * ThreadsPerWarp;

    // To achieve good memory coalescing patterns, we implement the following indexing:
    // reg[i:i+7] = weights[myL1Offset+PtxRegsPerThreadSize/8*i:myL1Offset+PtxRegsPerThreadSize/8*i+7]

    uint32_t instructionCount = 0, signal = 0;

#define SWITCH_REG(X) \
    switch (inst.decode_reg()) \
    { \
    case 0 : \
    case 1 : { \
        auto& r = regA; \
        X; \
        break; \
    } \
    case 2 : \
    case 3 : { \
        auto& r = regC; \
        X; \
        break; \
    } \
    default : \
        __builtin_unreachable(); \
    };

    constexpr int SharedMemoryBuckets = 4;

    __shared__ Eval::NNUE::L1Bucket bucketsShared[SharedMemoryBuckets];
    // bucketsShared[i - sharedBucketOffset], if in range, is buckets[i]
    int sharedBucketOffset = 8;

    __shared__ Instruction cmdBuffers[WarpsPerThreadBlock][MaxInstructionsCount];
    Instruction*           myCmdBuffer = cmdBuffers[warp_id % WarpsPerThreadBlock];

    while (true)
    {
        // Warp leader polls the queue
        if (lane_id == 0)
        {
            uint32_t temp;
            while ((temp = *(volatile uint32_t*) &instructionBuffer->data) == signal)
            {
                __nanosleep(50);  // TODO better approach here?
            }
            signal           = temp;
            instructionCount = signal & 0xffff;

            myCmdBuffer[instructionCount] = Instruction::nop();
        }

        __syncwarp();
        instructionCount = __shfl_sync(0xFFFFFFFF, instructionCount, 0);

        // Copy instructions into shared memory
        for (uint32_t i = lane_id; i < instructionCount; i += ThreadsPerWarp)
        {
            myCmdBuffer[i] = instructionBuffer->list[i];
        }

        __syncwarp();
        Instruction nextInst = myCmdBuffer[0];

        for (uint32_t inst_i = 0; inst_i < instructionCount; ++inst_i)
        {
            __syncwarp();

            Instruction inst = nextInst;
            nextInst         = myCmdBuffer[inst_i + 1];

            switch (inst.opcode())
            {
            case Nop :
                continue;
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
            case Exit : {
                if (lane_id == 0)
                {
                    machine->result[0] = 0;
                    __threadfence_system();
                }
                return;
            }
            case LdScratch : {
                int16_t* scratch = data->get_scratch(inst);
                SWITCH_REG({
                    _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                           i += vectorLoadStride, j += 4) {
                        int4 data = *(int4*) &scratch[i];
                        r[j]      = data.x;
                        r[j + 1]  = data.y;
                        r[j + 2]  = data.z;
                        r[j + 3]  = data.w;
                    }
                })
                break;
            }
            case StScratch : {
                int16_t* scratch = data->get_scratch(inst);
                SWITCH_REG({
                    _Pragma("unroll") for (int i = myL1Offset, j = 0; i < L1Size;
                                           i += vectorLoadStride, j += 4) {
                        int4 result;
                        result.x             = r[j];
                        result.y             = r[j + 1];
                        result.z             = r[j + 2];
                        result.w             = r[j + 3];
                        *(int4*) &scratch[i] = result;
                    }
                })

                break;
            }
            case Pack8 : {
                auto get = [&](int i) {
                    int j   = i % 2 ? 1 : 17;
                    int sum = regA[i / 2] + regC[i / 2];
                    return sum << j >> 17;
                };

                auto apply = [&](int p) {
                    int offset = p * (L1EntriesPerThreadSlice / 8);
                    for (int i = 0; i < L1EntriesPerThreadSlice / 8; ++i)
                        packed[offset + i] = 0;
#pragma unroll
                    for (int i = 0; i < L1EntriesPerThreadSlice / 2; ++i)
                    {
                        int sum0 = std::clamp(get(i), 0, 255);
                        int sum1 = std::clamp(get(i + L1EntriesPerThreadSlice / 2), 0, 255);

                        insert_byte(packed[offset + i / 4], unsigned(sum0 * sum1) / 512, i % 4);
                    }
                };

                if (inst.decode_pack_half())
                    apply(1);
                else
                    apply(0);
                break;
            }
            case AddFeature : {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t* weights = &transformer->weights[index * L1Size];
#pragma unroll
                    for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 4)
                    {
                        int4 data = *(int4*) &weights[i];
                        unpack16_to_32<Add>(data.x, regA[j]);
                        unpack16_to_32<Add>(data.y, regA[j + 1]);
                        unpack16_to_32<Add>(data.z, regA[j + 2]);
                        unpack16_to_32<Add>(data.w, regA[j + 3]);
                    }
                }
                else
                {
                    const int8_t* weights = &transformer->threatWeights[index * L1Size];
#pragma unroll
                    for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 4)
                    {
                        int2 data = *(int2*) &weights[i];
                        unpack8_to_32<Add>(data.x, regC[j], regC[j + 1]);
                        unpack8_to_32<Add>(data.y, regC[j + 2], regC[j + 3]);
                    }
                }
                break;
            }
            case SubFeature : {
                uint32_t index = inst.decode_wide_index();
                if (is_halfka_reg(inst.decode_reg()))
                {
                    const int16_t* weights = &transformer->weights[index * L1Size];
#pragma unroll
                    for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 4)
                    {
                        int4 data = *(int4*) &weights[i];
                        unpack16_to_32<Sub>(data.x, regA[j]);
                        unpack16_to_32<Sub>(data.y, regA[j + 1]);
                        unpack16_to_32<Sub>(data.z, regA[j + 2]);
                        unpack16_to_32<Sub>(data.w, regA[j + 3]);
                    }
                }
                else
                {
                    const int8_t* weights = &transformer->threatWeights[index * L1Size];
#pragma unroll
                    for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 4)
                    {
                        int2 data = *(int2*) &weights[i];
                        unpack8_to_32<Sub>(data.x, regC[j], regC[j + 1]);
                        unpack8_to_32<Sub>(data.y, regC[j + 2], regC[j + 3]);
                    }
                }
                break;
            }
            case Finalize : {
                int                   bucketIndex = inst.decode_bucket();
                Eval::NNUE::L1Bucket* bucket;

                unsigned sharedIndex = bucketIndex - sharedBucketOffset;
                if (sharedIndex < SharedMemoryBuckets)
                    bucket = &bucketsShared[sharedIndex];
                else
                    bucket = &buckets[bucketIndex];

                int accumulation = lane_id < 16 ? bucket->biases[lane_id] : 0;
                unsigned activeMask = lane_id < 16 ? 0xFFFF : 0xFFFF0000U;

#pragma unroll
                for (int j = 0, q = 0; j < L1EntriesPerThreadSlice / 4; j += 2)
                {
                    unsigned nnz1 = __ballot_sync(0xFFFFFFFF, packed[j]) & activeMask;
                    unsigned nnz2 = __ballot_sync(0xFFFFFFFF, packed[j + 1]) & activeMask;

                    auto process_nnz = [&] (unsigned& nnz, int q_offset)
                    {
                        int th_i = __ffs(nnz) - 1;
                        nnz &= nnz - 1;

                        int selected = __shfl_sync(activeMask, packed[j + q_offset], th_i);
                        accumulation =
                          __dp4a(selected,
                                 *((int*) &bucket->weights[64 * (q + q_offset + 2 * th_i)] + (lane_id % 16)),
                                 accumulation);
                    };

                    while (nnz1)
                        process_nnz(nnz1, 0);

                    while (nnz2)
                        process_nnz(nnz2, 1);

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
#pragma unroll
                    for (int i = myL1Offset, j = 0; i < L1Size; i += vectorLoadStride, j += 4)
                    {
                        int4 data = *(int4*) &transformer->biases.data()[i];
                        unpack16_to_32<Store>(data.x, regA[j + 0]);
                        unpack16_to_32<Store>(data.y, regA[j + 1]);
                        unpack16_to_32<Store>(data.z, regA[j + 2]);
                        unpack16_to_32<Store>(data.w, regA[j + 3]);
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < PtxRegsPerThreadSlice; i++)
                        regC[i] = 0;
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
    checkError(cudaMalloc(&data, sizeof(RegisterData)));
    checkError(cudaMemset(data, 0, sizeof(RegisterData)));

    this->weights  = weights;
    this->wcBuffer = wcInstructionBuffer;
}

RegisterMachine::~RegisterMachine() {
    cudaStreamDestroy((cudaStream_t) stream);
    cudaFree(data);
    data     = nullptr;
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
      cudaMemcpyAsync(&array, &data->regs[index], sizeof(array), cudaMemcpyDeviceToHost, stream));
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
    cudaFree(machines);
    machines  = nullptr;
    wcBuffers = nullptr;
}

void CudaContext::launch_persistent_kernel() {
    if (stream)
        return;

    checkError(cudaStreamCreate((cudaStream_t*) &stream));
    memset(wcBuffers, 0, machineCount * sizeof(InstructionBuffer));

    int num_warps         = machineCount;
    int threads_per_block = ThreadsPerWarp * WarpsPerThreadBlock;
    int num_blocks        = (num_warps * 32 + threads_per_block - 1) / threads_per_block;

    for (size_t i = 0; i < machineCount; i++)
    {
        machines[i].isActive = true;
    }

    persistent_kernel<<<num_blocks, threads_per_block, 0, (cudaStream_t) stream>>>(
      machines, wcBuffers, machineCount);
}
}
