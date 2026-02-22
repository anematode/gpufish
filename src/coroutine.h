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

#ifndef COROUTINE_H_INCLUDED
#define COROUTINE_H_INCLUDED

#include <cstddef>

/*

#if defined(__x86_64__) || defined(_M_X64)
    #define X86_64_COROUTINE_IMPL 1
#else
    #define X86_64_COROUTINE_IMPL 0
#endif

*/

#define X86_64_COROUTINE_IMPL 1 /* TODO: TEMPORARILY DISABLED */

#if !X86_64_COROUTINE_IMPL
    #include <ucontext.h>
#endif // !X86_64_COROUTINE_IMPL

namespace Stockfish {

// a void function that accepts an int argument.
using CoroutineFunction = void(int);

// a low-level system for managing coroutines, similar to the setcontext API.
class CoroutineContext {
   public:
    // default constructor does not initialize any fields.
    CoroutineContext();

    // saves the current context into this object.
    // this function must be called before set_entry_point for compatibility with setcontext,
    // but it is not guaranteed that this context is safe to use after this call.
    void init_from_current_context();

    // sets the parent context.
    // after the entry function returns, control will be
    // switched to parent context.
    void set_parent_context(CoroutineContext* parent);

    // provides a custom stack region for this context to use.
    // must be run before set_entry_point.
    void set_stack_region(void* stackPointer, size_t stackSize);

    // sets up this context to begin invoking a given CoroutineFunction when switched to.
    // the current context and stack region must be initialized beforehand,
    // via the init_from_current_context() and set_stack_region() methods.
    void set_entry_point(CoroutineFunction* func, int invocationArgument);

    // saves the current context into this object, and
    // switches to the target context.
    void switch_to(CoroutineContext& target);

   private:
#if X86_64_COROUTINE_IMPL
    CoroutineContext* parentContext;
    char* stack;
    CoroutineFunction* coroutineFunction;
    int functionArgument;
    void *instructionPointer;
#else  // normal setcontext implementation
    ucontext_t context;
#endif
};


}  // namespace Stockfish

#endif  // #ifndef COROUTINE_H_INCLUDED