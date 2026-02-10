//
// Created by toystory on 2/7/26.
//

#ifndef GPUFISH_GPU_OPCODES_H
#define GPUFISH_GPU_OPCODES_H

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <string>

#define ScratchRegCount 2048
#define L1Size 1024
#define MaxInstructionsCount 512
#define ThreadsPerWarp 32

namespace Stockfish::GPU
{
    enum Opcode
    {
        SwitchMachine = 0,
        // reg = [mem]
        LdScratch = 1,
        // [mem] = reg
        StScratch = 2,
        // reg += [feature mem]
        //    - if reg is A or B, the feature is a king buckets feature;
        //    - if reg is C or D, the feature is a threat weights feature.
        AddFeature = 3,
        // reg -= [feature mem]
        //    - like above
        SubFeature = 4,
        // output_buf = sparse_matrix_multiply(M[bucket], pairwise_fuse((A + C), (B + D)))
        Finalize = 5,
        // Exit instruction loop
        Exit = 6,
        // Fill register with biases
        ResetReg = 7,
    };

    enum Reg
    {
        A, B, C, D
    };

    // "Register machine" architecture:
    //    - Four registers A, B, C, D, each of L1 size; these are volatile between machine swaps
    //    - Scratch space, each of L1 size, indexed up to 1024 -> used as AccumulatorStack
    //    - 64-byte output buffer for L2, shared with host
    // Each worker is given a separately allocated register machine

    struct Instruction
    {
        uint32_t data;

        static constexpr size_t OpcodeBits = 3;
        static constexpr size_t DataBits = 13;
        static constexpr size_t MaxMachineIndex = 1 << DataBits;
        static constexpr size_t WideIndexBits = 18;
        static constexpr size_t RegIndexBits = 2;
        static constexpr size_t BucketBits = 3;
        static constexpr size_t MaxBucket = 8;

        std::string to_string() const
        {
            std::string result = "";
            switch (opcode())
            {
                case SwitchMachine: result = "SwitchMachine"; break;
                case LdScratch: result = "LdScratch #" + std::to_string(decode_wide_index()); break;
                case StScratch: result = "StScratch #" + std::to_string(decode_wide_index()); break;
                case AddFeature: result = "AddFeature #" + std::to_string(decode_wide_index()); break;
                case SubFeature: result = "SubFeature #" + std::to_string(decode_wide_index()); break;
                case Exit: result = "Exit"; break;
                default: result = "unknown"; break;
            }
            return result;
        }

        static constexpr Instruction switch_to_machine(size_t index)
        {
            assert(index < (1 << DataBits));
            return {  uint32_t((index << OpcodeBits) + SwitchMachine) };
        }

        static constexpr void check_wide_index([[maybe_unused]] size_t idx)
        {
            assert(idx < (1 << WideIndexBits));
        }

        static constexpr void check_reg([[maybe_unused]] size_t idx)
        {
            assert(idx < (1 << RegIndexBits));
        }

        static constexpr Instruction make_mem_reg(Opcode opcode, Reg reg, size_t scratchIdx)
        {
            check_wide_index(scratchIdx);
            check_reg(reg);
            return {
            uint32_t((scratchIdx << (OpcodeBits + RegIndexBits)) + (reg << OpcodeBits) + opcode)
            };
        }

        static constexpr Instruction load_scratch(Reg regDst, size_t scratchSrc)
        {
            return make_mem_reg(LdScratch, regDst, scratchSrc);
        }

        static constexpr Instruction store_scratch(size_t scratchDst, Reg regSrc)
        {
            return make_mem_reg(StScratch, regSrc, scratchDst);
        }

        static constexpr Instruction add_feature(Reg regDst, size_t featureIndex)
        {
            return make_mem_reg(AddFeature, regDst, featureIndex);
        }

        static constexpr Instruction sub_feature(Reg regDst, size_t featureIndex)
        {
            return make_mem_reg(SubFeature, regDst, featureIndex);
        }

        static constexpr Instruction reset_reg(Reg regDst)
        {
            return {
                uint32_t(ResetReg + (regDst << OpcodeBits))
            };
        }

        static constexpr Instruction finalize(size_t bucketIdx, bool stm)
        {
            assert(bucketIdx < MaxBucket);
            return {
                uint32_t((stm << (OpcodeBits + BucketBits)) + (bucketIdx << OpcodeBits) + Finalize)
            };
        }

        static constexpr Instruction stop()
        {
            return { Exit };
        }

        constexpr Opcode opcode() const
        {
            return Opcode(data & 7);
        }

        constexpr uint32_t machine_index() const
        {
            assert(opcode() == SwitchMachine);
            return data >> OpcodeBits;
        }

        constexpr Reg decode_reg() const
        {
            assert(opcode() == LdScratch || opcode() == StScratch || opcode() == AddFeature || opcode() == SubFeature || opcode() == ResetReg);
            return Reg(data >> OpcodeBits & 3);
        }

        constexpr uint32_t decode_wide_index() const
        {
            assert(opcode() == LdScratch || opcode() == StScratch || opcode() == AddFeature || opcode() == SubFeature);
            return uint32_t(data >> (OpcodeBits + RegIndexBits));
        }

        constexpr size_t decode_bucket() const
        {
            assert(opcode() == Finalize);
            return data >> OpcodeBits;
        }

        constexpr bool side_to_move() const
        {
            assert(opcode() == Finalize);
            return bool(data >> (OpcodeBits + BucketBits));
        }
    };
}

#endif //GPUFISH_GPU_OPCODES_H