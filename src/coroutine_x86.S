# coroutine_x86.S
# implements the context switching functions for coroutine.cpp
# when X86_64_COROUTINE_IMPL is enabled.

.globl coroutine_ctx_switch_no_save
.type coroutine_ctx_switch_no_save @function

.globl coroutine_ctx_switch_save
.type coroutine_ctx_switch_save @function

.section .text

/* CoroutineContext X86_64_COROUTINE_IMPL Fields:
 * CoroutineContext* parentContext;      // offset 0
 * char* stack;                          // offest 8
 * CoroutineFunction* coroutineFunction; // offset 16
 * int functionArgument;                 // offset 24
 * void *instructionPointer;             // offset 32
 */
 
/*
 * extern "C" __attribute__ ((no_callee_saved_registers))
 * void coroutine_ctx_switch_no_save(CoroutineContext* self, CoroutineContext* target);
 */
coroutine_ctx_switch_no_save:
    movq  8(%rsi), %rdx  # load target->stack into rdx
    movq 16(%rsi), %rbx  # load target->coroutineFunction into rbx
    movl 24(%rsi), %ecx  # load target->functionArgument into rcx
    test %rbx, %rbx
    jz 0f  # if already been started, just switch to it
        # otherwise, set up call stack for the target
        sub $16, %rdx
        movq %rsi, 8(%rdx)  # push CoroutineContext* target
        lea 2f(%rip), %rax
        movq %rax, 0(%rdx)  # push trampoline (ret addr)
        mov %rdx, %rax
        sub $8, %rdx
        movq %rax, 0(%rdx)  # push same stack addr (rbp)
        movq $0, 16(%rsi)  # store 0 into target->coroutineFunction
        movq %rbx, 32(%rsi)  # store rbx into target->instructionPointer
        movq  %rdx, 8(%rsi)  # store back into target->stack

    0:  # switch to the coroutine
        push %rbp  # save rbp on our stack
        movq %rsp, 8(%rdi)  # store rsp in this->stack
        lea 1f(%rip), %rax
        movq %rax, 32(%rdi)  # store continuation addr in this->instructionPointer
        movq 8(%rsi), %rdx
        movq %rdx, %rsp  # use target->stack as rsp
        pop %rbp  # pop rbp from the new stack
        movq %rcx, %rdi  # copy target->functionArgument into rdi
        movq 32(%rsi), %rbx
        jmp *%rbx  # jump to target->instructionPointer, start executing

    1:  # control flow returned from coroutine
        # resume execution like normal
        ret

    2:  # the return trampoline
        # jump to the ctx->parentContext, stored on the stack
        pop %rdi  # load current context pointer into rdi
        movq $0, 32(%rdi)  # safety measure: clear instructionPointer
        movq 0(%rdi), %rsi  # load parentContext into rsi
        movq 8(%rsi), %rdx  # load parentContext->stack into rdx
        movq %rdx, %rsp  # use parentContext->stack as rsp
        pop %rbp  # pop rbp from the parent stack
        movq 32(%rsi), %rbx  # load parentContext->instructionPointer into rbx
        jmp *%rbx  # jump to parentContext->instructionPointer, start executing

/*
 * extern "C" // normal calling convention
 * void coroutine_ctx_switch_save(CoroutineContext* self, CoroutineContext* target);
 */
coroutine_ctx_switch_save:
    # manually save rbx, r12, r13, r14, r15
    # before calling coroutine_ctx_switch_no_save
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    call coroutine_ctx_switch_no_save  # 16-byte aligned: 5 pushes + 1 call
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    ret
