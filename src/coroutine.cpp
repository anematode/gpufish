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

__attribute__ ((naked))
__attribute__ ((preserve_none))
void CoroutineContext::switch_to(CoroutineContext& target) {
    asm(
        "xor eax, eax;\n"
        "ret\n"
    );

    // if (target.coroutineFunction)
    // {
        // hasn't started executing target for the first time yet;
        // push &target
        // push ret address: trampoline
        // set up the call stack for target, and arg in a temp1 register
        // set target.instructionPointer = coroutineFunction
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