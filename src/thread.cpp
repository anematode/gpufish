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

#include "thread.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "bitboard.h"
#include "history.h"
#include "memory.h"
#include "movegen.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "timeman.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

#include <ucontext.h>

namespace Stockfish {

const int WorkersPerThread = 8;

// Constructor launches the thread and waits until it goes to sleep
// in idle_loop(). Note that 'searching' and 'exit' should be already set.
Thread::Thread(Search::SharedState&                    sharedState,
               std::unique_ptr<Search::ISearchManager> sm,
               size_t                                  n,
               size_t                                  numaN,
               size_t                                  totalNumaCount,
               OptionalThreadToNumaNodeBinder          binder) :
    idx(n),
    idxInNuma(numaN),
    totalNuma(totalNumaCount),
    nthreads(sharedState.options["Threads"]),
    stdThread(&Thread::idle_loop, this) {

    searchManager = std::move(sm);

    wait_for_search_finished();

    run_custom_job([this, &binder, &sharedState, n]() {
        // Use the binder to [maybe] bind the threads to a NUMA node before doing
        // the Worker allocation. Ideally we would also allocate the SearchManager
        // here, but that's minor.
        this->numaAccessToken = binder();

        for (int i = 0; i < WorkersPerThread; ++i)
        {
            this->workers.push_back(make_unique_large_page<Search::Worker>(
              sharedState, *searchManager, n, i, idxInNuma, totalNuma, this->numaAccessToken, this));
        }
    });

    wait_for_search_finished();
}


// Destructor wakes up the thread in idle_loop() and waits
// for its termination. Thread should be already waiting.
Thread::~Thread() {

    assert(!searching);

    exit = true;
    start_searching();
    stdThread.join();
}

thread_local Thread* curr_thread;

static void start_searching_fwd(int idx) {
    Search::Worker* worker = curr_thread->workers[idx].get();
    sf_assume(worker != nullptr);

    worker->start_searching();
    worker->is_active = false;

    // safety mitigation: save this context just in case the worker is invoked after it is done.
    // this should never happen, but may be useful for debugging the async scheduler.
    // can be removed for optimisation
    if (getcontext(&worker->activeContext) == -1)
    {
        perror("getcontext 2");
        abort();
    }
}

std::mutex mtx;

// Wakes up the thread that will start the search
void Thread::start_searching() {
    // assert(worker != nullptr);

    run_custom_job([this]() {
        ucontext_t main;

        curr_thread = this;
        for (size_t i = 0; i < workers.size(); ++i)
        {
            auto& context = workers.at(i)->activeContext;
            if (getcontext(&context) == -1)
            {
                perror("getcontext");
                abort();
            }
            auto& worker = workers.at(i);
            context.uc_link = &main;
            context.uc_stack.ss_size = worker->contextStack.size;
            context.uc_stack.ss_sp = worker->contextStack.mem;

            worker->is_active = true;
            worker->disable_yielding = false;

            makecontext(&worker->activeContext, reinterpret_cast<void(*)()>(&start_searching_fwd), 1, (int)i);
        }

        // invoke a worker
        if (swapcontext(&main, &workers[0].get()->activeContext) == -1)
        {
            perror("swapcontext 1");
            abort();
        }

        // Iterate over all workers and step all active ones to completion
        // todo: this should also call a method on Thread rather than rewriting this logic here.
        // todo: similar with sequence in Search::Worker::start_searching()
        for (auto & worker : workers)
        {
            worker->disable_yielding = true;
            if (!worker->is_active) continue;

            if (swapcontext(&main, &worker->activeContext) == -1)
            {
                perror("swapcontext 2");
                abort();
            }
        }
    });
}

// Clears the histories for the thread worker (usually before a new game)
void Thread::clear_worker() {
    //assert(worker != nullptr);
    run_custom_job([this]() {
        for (auto& worker : workers)
            worker->clear();
    });
}

// Blocks on the condition variable until the thread has finished searching
void Thread::wait_for_search_finished() {

    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
}

// Launching a function in the thread
void Thread::run_custom_job(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return !searching; });
        jobFunc   = std::move(f);
        searching = true;
    }
    cv.notify_one();
}

void Thread::ensure_network_replicated() {
    for (auto& worker : workers)
        worker->ensure_network_replicated();
}

// Thread gets parked here, blocked on the condition variable
// when the thread has no work to do.

void Thread::idle_loop() {
    while (true)
    {
        std::unique_lock<std::mutex> lk(mutex);
        searching = false;
        cv.notify_one();  // Wake up anyone waiting for search finished
        cv.wait(lk, [&] { return searching; });

        if (exit)
            return;

        std::function<void()> job = std::move(jobFunc);
        jobFunc                   = nullptr;

        lk.unlock();

        if (job)
            job();
    }
}

Search::SearchManager* ThreadPool::main_manager() {
    return static_cast<Search::SearchManager*>(main_thread()->searchManager.get());
}

uint64_t ThreadPool::nodes_searched() const { return accumulate(&Search::Worker::nodes); }
uint64_t ThreadPool::tb_hits() const { return accumulate(&Search::Worker::tbHits); }

static size_t next_power_of_two(uint64_t count) { return count > 1 ? (2ULL << msb(count - 1)) : 1; }

// Creates/destroys threads to match the requested number.
// Created and launched threads will immediately go to sleep in idle_loop.
// Upon resizing, threads are recreated to allow for binding if necessary.
void ThreadPool::set(const NumaConfig&                           numaConfig,
                     Search::SharedState                         sharedState,
                     const Search::SearchManager::UpdateContext& updateContext) {

    if (threads.size() > 0)  // destroy any existing thread(s)
    {
        main_thread()->wait_for_search_finished();

        threads.clear();

        boundThreadToNumaNode.clear();
    }

    const size_t requested = sharedState.options["Threads"];

    if (requested > 0)  // create new thread(s)
    {
        // Binding threads may be problematic when there's multiple NUMA nodes and
        // multiple Stockfish instances running. In particular, if each instance
        // runs a single thread then they would all be mapped to the first NUMA node.
        // This is undesirable, and so the default behaviour (i.e. when the user does not
        // change the NumaConfig UCI setting) is to not bind the threads to processors
        // unless we know for sure that we span NUMA nodes and replication is required.
        const std::string numaPolicy(sharedState.options["NumaPolicy"]);
        const bool        doBindThreads = [&]() {
            if (numaPolicy == "none")
                return false;

            if (numaPolicy == "auto")
                return numaConfig.suggests_binding_threads(requested);

            // numaPolicy == "system", or explicitly set by the user
            return true;
        }();

        std::map<NumaIndex, size_t> counts;
        boundThreadToNumaNode = doBindThreads
                                ? numaConfig.distribute_threads_among_numa_nodes(requested)
                                : std::vector<NumaIndex>{};

        if (boundThreadToNumaNode.empty())
            counts[0] = requested;  // Pretend all threads are part of numa node 0
        else
        {
            for (size_t i = 0; i < boundThreadToNumaNode.size(); ++i)
                counts[boundThreadToNumaNode[i]]++;
        }

        sharedState.sharedHistories.clear();
        for (auto pair : counts)
        {
            NumaIndex numaIndex = pair.first;
            uint64_t  count     = pair.second;
            auto      f         = [&]() {
                sharedState.sharedHistories.try_emplace(numaIndex, next_power_of_two(count));
            };
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaIndex, f);
            else
                f();
        }

        auto threadsPerNode = counts;
        counts.clear();

        while (threads.size() < requested)
        {
            const size_t    threadId      = threads.size();
            const NumaIndex numaId        = doBindThreads ? boundThreadToNumaNode[threadId] : 0;
            auto            create_thread = [&]() {
                auto manager = threadId == 0
                                          ? std::unique_ptr<Search::ISearchManager>(
                                   std::make_unique<Search::SearchManager>(updateContext))
                                          : std::make_unique<Search::NullSearchManager>();

                // When not binding threads we want to force all access to happen
                // from the same NUMA node, because in case of NUMA replicated memory
                // accesses we don't want to trash cache in case the threads get scheduled
                // on the same NUMA node.
                auto binder = doBindThreads ? OptionalThreadToNumaNodeBinder(numaConfig, numaId)
                                                       : OptionalThreadToNumaNodeBinder(numaId);

                threads.emplace_back(std::make_unique<Thread>(sharedState, std::move(manager),
                                                                         threadId, counts[numaId]++,
                                                                         threadsPerNode[numaId], binder));
            };

            // Ensure the worker thread inherits the intended NUMA affinity at creation.
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaId, create_thread);
            else
                create_thread();
        }

        clear();

        main_thread()->wait_for_search_finished();
    }
}


// Sets threadPool data to initial values
void ThreadPool::clear() {
    if (threads.size() == 0)
        return;

    for (auto&& th : threads)
        th->clear_worker();

    for (auto&& th : threads)
        th->wait_for_search_finished();

    // These two affect the time taken on the first move of a game:
    main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
    main_manager()->previousTimeReduction    = 0.85;

    main_manager()->callsCnt           = 0;
    main_manager()->bestPreviousScore  = VALUE_INFINITE;
    main_manager()->originalTimeAdjust = -1;
    main_manager()->tm.clear();
}

void ThreadPool::run_on_thread(size_t threadId, std::function<void()> f) {
    assert(threads.size() > threadId);
    threads[threadId]->run_custom_job(std::move(f));
}

void ThreadPool::wait_on_thread(size_t threadId) {
    assert(threads.size() > threadId);
    threads[threadId]->wait_for_search_finished();
}

size_t ThreadPool::num_threads() const { return threads.size(); }


// Wakes up main thread waiting in idle_loop() and returns immediately.
// Main thread will wake up other threads and start the search.
void ThreadPool::start_thinking(const OptionsMap&  options,
                                Position&          pos,
                                StateListPtr&      states,
                                Search::LimitsType limits) {

    main_thread()->wait_for_search_finished();

    main_manager()->stopOnPonderhit = stop = abortedSearch = false;
    main_manager()->ponder                                 = limits.ponderMode;

    increaseDepth = true;

    Search::RootMoves rootMoves;
    const auto        legalmoves = MoveList<LEGAL>(pos);

    for (const auto& uciMove : limits.searchmoves)
    {
        auto move = UCIEngine::to_move(pos, uciMove);

        if (std::find(legalmoves.begin(), legalmoves.end(), move) != legalmoves.end())
            rootMoves.emplace_back(move);
    }

    if (rootMoves.empty())
        for (const auto& m : legalmoves)
            rootMoves.emplace_back(m);

    Tablebases::Config tbConfig = Tablebases::rank_root_moves(options, pos, rootMoves);

    // After ownership transfer 'states' becomes empty, so if we stop the search
    // and call 'go' again without setting a new position states.get() == nullptr.
    assert(states.get() || setupStates.get());

    if (states.get())
        setupStates = std::move(states);  // Ownership transfer, states is now empty

    // We use Position::set() to set root position across threads. But there are
    // some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
    // be deduced from a fen string, so set() clears them and they are set from
    // setupStates->back() later. The rootState is per thread, earlier states are
    // shared since they are read-only.
    for (auto&& th : threads)
    {
        th->run_custom_job([&]() {
            for (auto& worker : th->workers)
            {
                worker->limits = limits;
                worker->nodes = worker->tbHits = worker->bestMoveChanges = 0;
                worker->nmpMinPly                                        = 0;
                worker->rootDepth = worker->completedDepth = 0;
                worker->rootMoves                          = rootMoves;
                worker->rootPos.set(pos.fen(), pos.is_chess960(), &worker->rootState);
                worker->rootState = setupStates->back();
                worker->tbConfig  = tbConfig;
            }
        });
    }

    for (auto&& th : threads)
        th->wait_for_search_finished();

    main_thread()->start_searching();
}

Search::Worker* ThreadPool::get_best_worker() const {

    Search::Worker* bestWorker = threads.front().get()->workers[0].get();
    Value           minScore   = VALUE_NONE;

    std::unordered_map<Move, int64_t, Move::MoveHash> votes(2 * size());

    // Find the minimum score of all threads
    for (auto&& th : threads)
    {
        for (auto& worker : th->workers)
        {
            minScore = std::min(minScore, worker->rootMoves[0].score);
        }
    }

    // Vote according to score and depth, and select the best thread
    auto thread_voting_value = [minScore](Search::Worker* worker) {
        return (worker->rootMoves[0].score - minScore + 14) * int(worker->completedDepth);
    };

    for (auto&& th : threads)
    {
        for (auto& worker : th->workers)
        {
            votes[worker->rootMoves[0].pv[0]] += thread_voting_value(worker.get());
        }
    }

    for (auto&& th : threads)
    {
        for (auto& worker : th->workers)
        {
            const auto bestThreadScore = bestWorker->rootMoves[0].score;
            const auto newThreadScore  = worker->rootMoves[0].score;

            const auto& bestThreadPV = bestWorker->rootMoves[0].pv;
            const auto& newThreadPV  = worker->rootMoves[0].pv;

            const auto bestThreadMoveVote = votes[bestThreadPV[0]];
            const auto newThreadMoveVote  = votes[newThreadPV[0]];

            const bool bestThreadInProvenWin = is_win(bestThreadScore);
            const bool newThreadInProvenWin  = is_win(newThreadScore);

            const bool bestThreadInProvenLoss =
              bestThreadScore != -VALUE_INFINITE && is_loss(bestThreadScore);
            const bool newThreadInProvenLoss =
              newThreadScore != -VALUE_INFINITE && is_loss(newThreadScore);

            // We make sure not to pick a thread with truncated principal variation
            const bool betterVotingValue =
              thread_voting_value(worker.get()) * int(newThreadPV.size() > 2)
              > thread_voting_value(bestWorker) * int(bestThreadPV.size() > 2);

            if (bestThreadInProvenWin)
            {
                // Make sure we pick the shortest mate / TB conversion
                if (newThreadScore > bestThreadScore)
                    bestWorker = worker.get();
            }
            else if (bestThreadInProvenLoss)
            {
                // Make sure we pick the shortest mated / TB conversion
                if (newThreadInProvenLoss && newThreadScore < bestThreadScore)
                    bestWorker = worker.get();
            }
            else if (newThreadInProvenWin || newThreadInProvenLoss
                     || (!is_loss(newThreadScore)
                         && (newThreadMoveVote > bestThreadMoveVote
                             || (newThreadMoveVote == bestThreadMoveVote && betterVotingValue))))
                bestWorker = worker.get();
        }
    }

    return bestWorker;
}


// Start non-main threads.
// Will be invoked by main thread after it has started searching.
void ThreadPool::start_searching() {

    for (auto&& th : threads)
        if (th != threads.front())
            th->start_searching();
}


// Wait for non-main threads
void ThreadPool::wait_for_search_finished() const {

    for (auto&& th : threads)
        if (th != threads.front())
            th->wait_for_search_finished();
}

std::vector<size_t> ThreadPool::get_bound_thread_count_by_numa_node() const {
    std::vector<size_t> counts;

    if (!boundThreadToNumaNode.empty())
    {
        NumaIndex highestNumaNode = 0;
        for (NumaIndex n : boundThreadToNumaNode)
            if (n > highestNumaNode)
                highestNumaNode = n;

        counts.resize(highestNumaNode + 1, 0);

        for (NumaIndex n : boundThreadToNumaNode)
            counts[n] += 1;
    }

    return counts;
}

void ThreadPool::ensure_network_replicated() {
    for (auto&& th : threads)
        th->ensure_network_replicated();
}

}  // namespace Stockfish
