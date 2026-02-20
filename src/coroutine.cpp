/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "coroutine.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>

namespace Stockfish {
#if X86_64_COROUTINE_IMPL /* TODO: INCOMPLETE */

CoroutineContext::CoroutineContext() {
    parentContext = nullptr;
    stack = nullptr;
    coroutineFunction = nullptr;
    instructionPointer = nullptr;
}

void CoroutineContext::init_from_current_context() {}

void CoroutineContext::set_parent_context(CoroutineContext* parent) {
    parentContext = parent;
}

void CoroutineContext::set_stack_region(void* stackPointer, size_t stackSize) {
    stack = reinterpret_cast<char*>(stackPointer) + stackSize;
}

void CoroutineContext::set_entry_point(CoroutineFunction* func, int invocationArgument) {
    coroutineFunction = func;
    functionArgument = invocationArgument;
}

/* CoroutineContext X86_64_COROUTINE_IMPL Fields:
 * CoroutineContext* parentContext;      // offset 0
 * char* stack;                          // offest 8
 * CoroutineFunction* coroutineFunction; // offset 16
 * int functionArgument;                 // offset 24
 * void *instructionPointer;             // offset 32
 */

__attribute__ ((naked))
// __attribute__ ((preserve_none))
__attribute__ ((no_callee_saved_registers))
void CoroutineContext::switch_to(CoroutineContext& target) {
    // ABI:
    // rdi;  CoroutineContext* this
    // rsi;  CoroutineContext* target
    asm volatile (
        "movq  8(%rsi), %rdx; \n" // load target->stack into rdx
        "movq 16(%rsi), %rbx; \n" // load target->coroutineFunction into rbx
        "movl 24(%rsi), %ecx; \n" // load target->functionArgument into rcx
        "test %rbx, %rbx; \n"
        "jz 0f; \n" // if already been started, just switch to it
            // otherwise, set up call stack for the target
            "sub $16, %rdx; \n"
            "movq %rsi, 0(%rdx); \n" // push CoroutineContext* target
            "lea 2f(%rip), %rax; \n"
            "movq %rax, 8(%rdx); \n" // push trampoline (ret addr)
            "mov %rdx, %rax; \n"
            "sub $8, %rdx; \n"
            "movq %rax, 8(%rdx); \n" // push same stack addr (rbp)
            "movq $0, 16(%rsi); \n" // store 0 into target->coroutineFunction
            "movq %rbx, 32(%rsi); \n" // store rbx into target->instructionPointer
        "0: \n" // switch to the coroutine
            "push %rbp; \n" // save rbp on our stack
            "movq %rsp, 8(%rdi); \n" // store rsp in this->stack
            "lea 1f(%rip), %rax; \n"
            "movq %rax, 32(%rdi); \n" // store continuation addr in this->instructionPointer
            "movq %rdx, %rsp; \n" // use target->stack as rsp
            "pop %rbp; \n" // pop rbp from the new stack
            "movq %rcx, %rdi; \n" // copy target->functionArgument into rdi
            "movq 32(%rsi), %rbx; \n"
            "jmp *%rbx; \n"  // jump to target->instructionPointer, start executing
        "1: \n" // control flow returned from coroutine
            // resume execution like normal
            "pop %rbp; \n" // pop rbp from our stack
            "ret; \n"
        "2: \n" // the return trampoline
            // jump to the CoroutineContext* parentContext, stored on the stack
            "pop %rsi; \n" // load parentContext into rsi
            "movq 32(%rsi), %rbx; \n"
            "jmp *%rbx; \n"  // jump to parentContext->instructionPointer, start executing
    );

    // if (target.coroutineFunction)
    // {
        // hasn't started executing target for the first time yet;
        // on target stack
        // push &target
        // push ret address: trampoline
        // push same stack addr: rbp
        // set up the call stack for target, and arg in a temp1 register
        // set target.instructionPointer = target.coroutineFunction
        // set target.coroutineFunction to nullptr
    // }

    // load target.instructionPointer into a temp2 register
    // push rbp to our stack
    // save our rsp to this.stack
    // save our rip to this.instructionPointer
    // set stack to target.stack
    // pop rbp
    // put temp1 value into rdi register (call arg)
    // start executing temp2 as rip

    // trampoline(CoroutineContext* retContext /* on stack */) {
    //     // load the retContext stack again and jump to it
    // }
}

#else  // normal setcontext implementation

CoroutineContext::CoroutineContext() {}

void CoroutineContext::init_from_current_context() {
    if (getcontext(&context) == -1)
    {
        perror("getcontext");
        abort();
    }
}

void CoroutineContext::set_parent_context(CoroutineContext* parent) {
    context.uc_link = &parent->context;
}

void CoroutineContext::set_stack_region(void* stackPointer, size_t stackSize) {
    context.uc_stack.ss_size = stackSize;
    context.uc_stack.ss_sp   = stackPointer;
}

void CoroutineContext::set_entry_point(CoroutineFunction* func, int invocationArgument) {
    makecontext(&context, reinterpret_cast<void (*)()>(func), 1, (int) invocationArgument);
}

void CoroutineContext::switch_to(CoroutineContext& target) {
    if (swapcontext(&context, &target.context) == -1)
    {
        perror("swapcontext");
        abort();
    }
}

#endif

}  // namespace Stockfish