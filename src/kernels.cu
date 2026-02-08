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

    constexpr int RegI16Count = L1Size / ThreadsPerWarp;
    constexpr int RegCount = RegI16Count / 2;  // each unsigned contains two 16-bit values

    struct ScratchReg
    {
        int16_t data[L1Size];  // TODO: psqt?
    };

    // Device-side data that lives only in GPU memory
    struct RegisterData
    {
        ScratchReg regs[ScratchRegCount];

        int16_t *get(Instruction inst, uint32_t lane_id)
        {
            return regs[inst.decode_wide_index()].data + lane_id * RegI16Count;
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
    };

    class CudaContext
    {
    public:
        RegisterMachine *machines;
        size_t machineCount;
        WeightsData weights;

        CudaContext(const Eval::NNUE::NetworkBig& big, size_t machineCount) : machineCount(machineCount), weights(big)
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

        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        ~CudaContext()
        {
            for (int i = 0; i < machineCount; i++)
            {
                cudaFreeHost(machines[i].data);
            }
            cudaFree(machines);
            machines = nullptr;
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

    __global__ void persistent_kernel(RegisterMachine* machines, int num_machines) {
        unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerWarp;
        unsigned lane_id = threadIdx.x % ThreadsPerWarp;

        // Each warp picks a queue to monitor
        if (warp_id >= num_machines) return;

        RegisterMachine *machine = &machines[warp_id];
        RegisterData *data = machine->data;

        typedef unsigned reg_t[RegCount];

        reg_t regA, regB, regC, regD;

#define FOR_EACH_REG(X) switch (inst.decode_reg()) { \
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
            case LdScratch:
            {
                break;
            }
            case StScratch:
            {
                int16_t* scratch = data->get(inst, lane_id);

            }
            case AddFeature:
                {
                const int16_t* scratch = data->get(inst, lane_id);
                if (inst.decode_reg().)
#define X(reg) _Pragma("unroll") for (int i = 0; i < RegCount; i++) { \
                    unsigned val; \
                    memcpy(&val, &scratch[2 * i], 4); \
                    reg[i] = __vadd2(reg[i], val); \
                }
                FOR_EACH_REG(X)
                break;
            }
            case SubFeature:
                break;
            case ComputeL1:
                break;
            case ZeroReg:
                {
#undef X
#define X(reg) _Pragma("unroll") for (int i = 0; i < RegCount; i++) reg[i] = 0;
                    FOR_EACH_REG(X)

                    break;
                }
            }

            // 2. Execution (Warp-parallel)
            if (inst.opcode == OP_ADD) {
                for (int k = lane_id; k < cols; k += 32) {
                    matrix[inst.i * cols + k] += matrix[inst.j * cols + k];
                }
            } else if (inst.opcode == OP_READSUM) {
                float sum = 0;
                for (int k = lane_id; k < cols; k += 32) {
                    sum += matrix[inst.i * cols + k];
                }
                // Warp reduction using shuffles
                for (int offset = 16; offset > 0; offset /= 2)
                    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

                if (lane_id == 0) {
                    q->commands[q->tail].result = sum;
                    __threadfence_system();
                    q->commands[q->tail].status = DONE;
                }
            }

            if (lane_id == 0) {
                machine->tail = (machine->tail + 1) % QUEUE_SIZE;
            }
        }
    }

    std::unique_ptr<CudaContext> make_context()
    {

    }
}
