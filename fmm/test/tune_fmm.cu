/*
 * FMM parameter tuning benchmark for GPU dual traversal
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Parameter tuning benchmark for GPU FMM dual traversal
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Sweeps over numWarps and TraversalConfig parameters using REAL spherical
 * multipole M2L and P2P workloads (4th-order expansion) rather than the
 * synthetic fmaf P2P loop in tune.cu.
 *
 * The dual traversal kernel is run for each config; L2L downsweep + L2P are
 * excluded from timing since they are constant across configs.
 *
 * Benchmark protocol: 1 validation run (checking M2L/P2P counts and energy),
 * 3 warmup runs, then 10 timed runs. Reports timing, interaction counts,
 * energy error, queue diagnostics, and speedup vs production config.
 */

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <utility>
#include <ctime>
#include <chrono>
#include <sys/stat.h>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/traversal_gpu.cuh"
#include "cstone/traversal/groups_gpu.h"

#include "coord_samples/random.hpp"
#include "performance/timing.cuh"

#include "ryoanji/interface/treebuilder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/upsweep_gpu.h"

#include "fmm/spherical_multipole.cuh"

using namespace cstone;

namespace
{

// ════════════════════════════════════════════════════════════════════════════════
//  USER-EDITABLE TUNING RANGES
// ════════════════════════════════════════════════════════════════════════════════

//  numWarps values to benchmark (consumer warps per block = numWarps - producerWarpsPerBlock).
using TuneWarpList = std::integer_sequence<int, 6, 7, 8>;

//  blocksPerCluster — fixed for all configurations:
static constexpr unsigned kBpc = 8;

//  TraversalConfig parameter set.
struct TuneParams
{
    unsigned stackCap;
    unsigned chunkSize;
    unsigned forcePush;
    unsigned attemptPush;
    unsigned attemptPop;
    unsigned forcePop;
    unsigned travChunkSize;
    unsigned travForcePush;
    unsigned travAttemptPush;
    unsigned travAttemptPop;
};

// static constexpr TuneParams withChunkSize      (TuneParams p, unsigned v) { p.chunkSize       = v; return p; }
// static constexpr TuneParams withForcePush      (TuneParams p, unsigned v) { p.forcePush       = v; return p; }
// static constexpr TuneParams withAttemptPush    (TuneParams p, unsigned v) { p.attemptPush     = v; return p; }
// static constexpr TuneParams withAttemptPop     (TuneParams p, unsigned v) { p.attemptPop      = v; return p; }
// static constexpr TuneParams withForcePop       (TuneParams p, unsigned v) { p.forcePop        = v; return p; }
// static constexpr TuneParams withTravChunkSize  (TuneParams p, unsigned v) { p.travChunkSize   = v; return p; }
// static constexpr TuneParams withTravForcePush  (TuneParams p, unsigned v) { p.travForcePush   = v; return p; }
// static constexpr TuneParams withTravAttemptPush(TuneParams p, unsigned v) { p.travAttemptPush = v; return p; }
// static constexpr TuneParams withTravAttemptPop (TuneParams p, unsigned v) { p.travAttemptPop  = v; return p; }

template<size_t MaxConfigs>
struct TuneConfigBuilder
{
    std::array<TuneParams, MaxConfigs> values{};
    size_t count = 0;
};

constexpr size_t kMaxTuneConfigs = 50;

struct BuiltTuneConfigs
{
    std::array<TuneParams, kMaxTuneConfigs> values{};
    size_t count = 0;
};

consteval bool sameTuneParams(const TuneParams& a, const TuneParams& b)
{
    return a.stackCap == b.stackCap &&
           a.chunkSize == b.chunkSize &&
           a.forcePush == b.forcePush &&
           a.attemptPush == b.attemptPush &&
           a.attemptPop == b.attemptPop &&
           a.forcePop == b.forcePop &&
           a.travChunkSize == b.travChunkSize &&
           a.travForcePush == b.travForcePush &&
           a.travAttemptPush == b.travAttemptPush &&
           a.travAttemptPop == b.travAttemptPop;
}

consteval bool plausibleTuneConfig(const TuneParams& p)
{
    if (p.chunkSize == 0) return false;
    if (p.forcePop > p.attemptPop) return false;
    if (p.attemptPop >= p.attemptPush) return false;
    if (p.attemptPush > p.forcePush) return false;
    if (p.attemptPush - p.attemptPop < p.chunkSize) return false;
    if (p.forcePush >= p.stackCap) return false;

    if (p.travChunkSize == 0) return false;
    if (p.travAttemptPop > p.travAttemptPush) return false;
    if (p.travAttemptPush - p.travAttemptPop < p.travChunkSize) return false;
    if (p.travAttemptPush > p.travForcePush) return false;
    if (p.travForcePush >= p.stackCap) return false;
    return true;
}

template<size_t MaxConfigs>
consteval void pushUnique(TuneConfigBuilder<MaxConfigs>& b, const TuneParams& p)
{
    if (!plausibleTuneConfig(p)) return;
    for (size_t i = 0; i < b.count; ++i)
    {
        if (sameTuneParams(b.values[i], p)) return;
    }
    if (b.count < MaxConfigs) { b.values[b.count++] = p; }
}

// ~25 configs: 3 interaction seeds x ~10 traversal variants
consteval BuiltTuneConfigs buildFmmTuneConfigs()
{
    TuneConfigBuilder<kMaxTuneConfigs> b{};

    constexpr unsigned sc = 1024;

    // 3 interaction seeds bracketing production {sc, 224, 704, 512, 160, 96}
    constexpr std::array<TuneParams, 3> interactionSeeds{
        TuneParams{sc, 160, 608, 416, 128, 96, 0, 0, 0, 0}, // smaller
        TuneParams{sc, 224, 704, 512, 160, 96, 0, 0, 0, 0}, // production
        TuneParams{sc, 256, 768, 576, 192, 96, 0, 0, 0, 0}  // larger
    };

    struct TravVariant
    {
        unsigned chunkSize;
        unsigned forcePush;
        unsigned attemptPush;
        unsigned attemptPop;
    };

    // 10 traversal variants bracketing production {192, 640, 224, 32}
    constexpr std::array<TravVariant, 10> traversalVariants{{
        {192, 576, 192, 16},
        {192, 640, 224, 32},  // production
        {192, 704, 256, 32},
        {192, 768, 224, 16},
        {192, 640, 256, 48},
        {192, 576, 224, 32},
        {192, 768, 256, 32},
        {192, 896, 224, 16},
        {256, 768, 320, 64},
        {160, 640, 224, 32}
    }};

    for (const auto& iSeed : interactionSeeds)
    {
        for (const auto& tVar : traversalVariants)
        {
            TuneParams p    = iSeed;
            p.travChunkSize   = tVar.chunkSize;
            p.travForcePush   = tVar.forcePush;
            p.travAttemptPush = tVar.attemptPush;
            p.travAttemptPop  = tVar.attemptPop;
            pushUnique(b, p);
        }
    }

    BuiltTuneConfigs out{};
    out.count = b.count;
    for (size_t i = 0; i < b.count; ++i)
        out.values[i] = b.values[i];
    return out;
}

static constexpr BuiltTuneConfigs builtTuneConfigs = buildFmmTuneConfigs();
static constexpr auto& tuneConfigs                 = builtTuneConfigs.values;
static constexpr size_t numTuneConfigs             = builtTuneConfigs.count;
static_assert(numTuneConfigs <= 50, "Per-warp configuration budget exceeded (max 50)");
static_assert(numTuneConfigs * TuneWarpList::size() <= 300,
              "Configured sweep exceeds 300 total benchmarks");

// ════════════════════════════════════════════════════════════════════════════════
//  END OF USER-EDITABLE SECTION
// ════════════════════════════════════════════════════════════════════════════════

// ── Statistics ────────────────────────────────────────────────────────────────

struct TuneStats
{
    float median, mean, stddev, minVal, maxVal;
};

static TuneStats computeStats(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    size_t n   = v.size();
    float  med = (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    float  sum = std::accumulate(v.begin(), v.end(), 0.f);
    float  avg = sum / float(n);
    float  sq  = 0.f;
    for (float s : v)
        sq += (s - avg) * (s - avg);
    return {med, avg, std::sqrt(sq / float(n)), v.front(), v.back()};
}

static void printDual(FILE* outFile, const char* fmt, ...)
{
    char    buffer[4096];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    std::printf("%s", buffer);
    if (outFile) { std::fputs(buffer, outFile); }
}

// ── Result record ─────────────────────────────────────────────────────────────

struct FmmTuneResult
{
    int        numWarps;
    TuneParams params;
    int        totalBlocks;
    bool       valid;
    unsigned   m2lCount, p2pCount;
    unsigned   iactWHead, iactRHead;
    unsigned   travWHead, travRHead;
    unsigned   iactSegReadyBusy, iactSegCountBusy, travSegReadyBusy;
    double     energy;
    double     energyError;
    bool       countsMismatch;
    TuneStats  stats;
};

// ── Compile-time mapping: TuneParams[I] -> TraversalConfig<...> ──────────────

template<size_t I, int NW>
using FmmTuneTravConfig = TraversalConfig<tuneConfigs[I].stackCap,
                                          tuneConfigs[I].chunkSize,
                                          tuneConfigs[I].forcePush,
                                          tuneConfigs[I].attemptPush,
                                          tuneConfigs[I].attemptPop,
                                          tuneConfigs[I].forcePop,
                                          tuneConfigs[I].travChunkSize,
                                          tuneConfigs[I].travForcePush,
                                          tuneConfigs[I].travAttemptPush,
                                          tuneConfigs[I].travAttemptPop,
                                          (NW > 5) ? 2 : 1>;

// ── FMM dual traversal kernel with TravConfig as template parameter ──────────

template<int numWarps, class TravConfig, class T>
__global__ void tuneFmmDualTraversalKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const ryoanji::Vec3<T>* __restrict__ geoCenters,
    const ryoanji::Vec3<T>* __restrict__ geoSizes,
    const ryoanji::Vec4<T>* __restrict__ centers,
    const fmm::SphericalMultipole<T>* __restrict__ multipoles,
    fmm::SphericalLocalExpansion<T>* __restrict__ locals,
    const TreeNodeIndex* __restrict__ internalToLeaf,
    const ryoanji::LocalIndex* __restrict__ layout,
    const T* __restrict__ x,
    const T* __restrict__ y,
    const T* __restrict__ z,
    const T* __restrict__ h,
    const T* __restrict__ m,
    T* __restrict__ ppot,
    T* __restrict__ pax,
    T* __restrict__ pay,
    T* __restrict__ paz,
    ryoanji::LocalIndex firstTarget,
    fmm::GpuSphericalTables tables,
    GlobalWorkQueue gq,
    GlobalTraversalQueue tq,
    unsigned* nProd,
    unsigned* d_m2lCount,
    unsigned* d_p2pCount)
{
    using ryoanji::Vec3;
    using ryoanji::Vec4;
    using ryoanji::LocalIndex;

    auto continuation = [centers, geoCenters, geoSizes]
        __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        return cstone::evaluateMac(util::makeVec3(centers[a]), centers[a][3],
                                    geoCenters[b], geoSizes[b]);
    };

    auto m2l = [centers, multipoles, locals, tables, d_m2lCount] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        atomicAdd(d_m2lCount, 1u);
        fmm::SphericalLocalExpansion<T> localBuf;
        for (auto& v : localBuf)
            v = fmm::Complex<T>(0, 0);
        fmm::M2L(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], localBuf, tables.prefactor,
            tables.Anm, tables.Cnm);
        for (int i = 0; i < fmm::Nterm<fmm::ExpansionOrder>; ++i)
        {
            atomicAdd(reinterpret_cast<T*>(&locals[a][i]), localBuf[i].real());
            atomicAdd(reinterpret_cast<T*>(&locals[a][i]) + 1, localBuf[i].imag());
        }
    };

    auto p2p = [internalToLeaf, layout, x, y, z, h, m, ppot, pax, pay, paz, firstTarget,
                d_p2pCount] __device__(unsigned p2pMask, TreeNodeIndex a, TreeNodeIndex b)
    {
        unsigned lane = threadIdx.x % cstone::GpuConfig::warpSize;
        if (!((p2pMask >> lane) & 1u)) return;
        atomicAdd(d_p2pCount, 1u);
        TreeNodeIndex aLeaf = internalToLeaf[a];
        TreeNodeIndex bLeaf = internalToLeaf[b];

        LocalIndex aFirst = layout[aLeaf];
        LocalIndex aLast  = layout[aLeaf + 1];
        LocalIndex bFirst = layout[bLeaf];
        LocalIndex bLast  = layout[bLeaf + 1];

        for (LocalIndex t = aFirst; t < aLast; ++t)
        {
            Vec4<T> acc{0, 0, 0, 0};
            Vec3<T> target{x[t], y[t], z[t]};
            for (LocalIndex s = bFirst; s < bLast; ++s)
            {
                acc = ryoanji::P2P(acc, target, Vec3<T>{x[s], y[s], z[s]}, m[s], h[t], h[s]);
            }
            LocalIndex ti = t - firstTarget;
            atomicAdd(&ppot[ti], acc[0]);
            atomicAdd(&pax[ti], acc[1]);
            atomicAdd(&pay[ti], acc[2]);
            atomicAdd(&paz[ti], acc[3]);
        }
    };

    dualTraversalGPU<numWarps, TravConfig>(childOffsets, TreeNodeIndex(0), TreeNodeIndex(0), gq, tq, nProd,
                                           continuation, m2l, p2p);
}

// ── Benchmark one (numWarps, TravConfig) combination ─────────────────────────

template<int numWarps, class TravConfig, class T>
FmmTuneResult benchOneFmm(
    const TuneParams& params,
    // Tree arrays on device
    TreeNodeIndex* d_childOffsets,
    ryoanji::Vec3<T>* d_geoCenters,
    ryoanji::Vec3<T>* d_geoSizes,
    ryoanji::Vec4<T>* d_centers,
    fmm::SphericalMultipole<T>* d_multipoles,
    fmm::SphericalLocalExpansion<T>* d_locals,
    TreeNodeIndex* d_internalToLeaf,
    TreeNodeIndex* d_leafToInternal,
    ryoanji::LocalIndex* d_layout,
    // Particle arrays on device
    T* d_x, T* d_y, T* d_z, T* d_h, T* d_m,
    T* d_ppot, T* d_pax, T* d_pay, T* d_paz,
    // Parameters
    ryoanji::LocalIndex firstTarget,
    ryoanji::LocalIndex numTargets,
    TreeNodeIndex numNodes,
    TreeNodeIndex firstLeafIndex,
    TreeNodeIndex lastLeafIndex,
    fmm::GpuSphericalTables tables,
    std::span<const TreeNodeIndex> levelRange,
    // CPU reference
    unsigned refM2l, unsigned refP2p, double refEnergy,
    const T* refMasses, float G,
    // Benchmark params
    unsigned nWarm, unsigned nRuns)
{
    using ryoanji::Vec3;
    using ryoanji::Vec4;
    using ryoanji::LocalIndex;

    constexpr unsigned tpb = numWarps * GpuConfig::warpSize;
    FmmTuneResult res{};
    res.numWarps = numWarps;
    res.params   = params;
    res.valid    = false;

    unsigned total   = kBpc * 64u;
    res.totalBlocks  = int(total);

    // ── Allocate global interaction queue ──
    constexpr unsigned gChunk = TravConfig::chunkSize;
    constexpr unsigned gSegs  = 2048;
    constexpr unsigned gCap   = gSegs * gChunk;

    TreeNodeIndex *d_gA, *d_gB;
    int*      d_gIsP2P;
    unsigned *d_wH, *d_rH, *d_segCount, *d_sR, *d_nP;
    cudaMalloc(&d_gA, gCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_gB, gCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_gIsP2P, gCap * sizeof(int));
    cudaMalloc(&d_wH, sizeof(unsigned));
    cudaMalloc(&d_rH, sizeof(unsigned));
    cudaMalloc(&d_segCount, gSegs * sizeof(unsigned));
    cudaMalloc(&d_sR, gSegs * sizeof(unsigned));
    cudaMalloc(&d_nP, sizeof(unsigned));
    GlobalWorkQueue gq{d_gA, d_gB, d_gIsP2P, d_wH, d_rH, d_segCount, d_sR, gSegs};

    // ── Allocate global traversal queue ──
    constexpr unsigned tChunk = TravConfig::travChunkSize;
    constexpr unsigned tSegs  = 2048;
    constexpr unsigned tCap   = tSegs * tChunk;

    TreeNodeIndex *d_tA, *d_tB;
    unsigned *d_twH, *d_trH, *d_tsR;
    cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex));
    cudaMalloc(&d_twH, sizeof(unsigned));
    cudaMalloc(&d_trH, sizeof(unsigned));
    cudaMalloc(&d_tsR, tSegs * sizeof(unsigned));
    GlobalTraversalQueue tq{d_tA, d_tB, d_twH, d_trH, d_tsR, tSegs};

    // ── Interaction counters ──
    unsigned *d_m2lCount, *d_p2pCount;
    cudaMalloc(&d_m2lCount, sizeof(unsigned));
    cudaMalloc(&d_p2pCount, sizeof(unsigned));

    // ── Launch config ──
    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = {total, 1, 1};
    cfg.blockDim = {tpb, 1, 1};

    cudaLaunchAttribute attr{};
    attr.id             = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {kBpc, 1, 1};
    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    // Reset lambda: zeros locals, accumulators, counters, queues
    auto resetAll = [&]()
    {
        cudaMemset(d_locals, 0, numNodes * sizeof(fmm::SphericalLocalExpansion<T>));
        cudaMemset(d_ppot, 0, numTargets * sizeof(T));
        cudaMemset(d_pax, 0, numTargets * sizeof(T));
        cudaMemset(d_pay, 0, numTargets * sizeof(T));
        cudaMemset(d_paz, 0, numTargets * sizeof(T));
        cudaMemset(d_m2lCount, 0, sizeof(unsigned));
        cudaMemset(d_p2pCount, 0, sizeof(unsigned));
        cudaMemset(d_wH, 0, sizeof(unsigned));
        cudaMemset(d_rH, 0, sizeof(unsigned));
        cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_sR, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_twH, 0, sizeof(unsigned));
        cudaMemset(d_trH, 0, sizeof(unsigned));
        cudaMemset(d_tsR, 0, tSegs * sizeof(unsigned));
        cudaMemset(d_nP, 0, sizeof(unsigned));
    };

    // Kernel-only launch lambda (no L2L/L2P)
    auto launchKernel = [&]()
    {
        cudaLaunchKernelEx(&cfg, tuneFmmDualTraversalKernel<numWarps, TravConfig, T>,
                           d_childOffsets, d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals,
                           d_internalToLeaf, d_layout, d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz,
                           firstTarget, tables, gq, tq, d_nP, d_m2lCount, d_p2pCount);
    };

    // Reset-then-launch lambda for kernel-only timing
    auto resetAndLaunchKernel = [&]()
    {
        // Reset only what changes between kernel runs (locals, accumulators, counters, queues)
        cudaMemset(d_locals, 0, numNodes * sizeof(fmm::SphericalLocalExpansion<T>));
        cudaMemset(d_ppot, 0, numTargets * sizeof(T));
        cudaMemset(d_pax, 0, numTargets * sizeof(T));
        cudaMemset(d_pay, 0, numTargets * sizeof(T));
        cudaMemset(d_paz, 0, numTargets * sizeof(T));
        cudaMemset(d_m2lCount, 0, sizeof(unsigned));
        cudaMemset(d_p2pCount, 0, sizeof(unsigned));
        cudaMemset(d_wH, 0, sizeof(unsigned));
        cudaMemset(d_rH, 0, sizeof(unsigned));
        cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_sR, 0, gSegs * sizeof(unsigned));
        cudaMemset(d_twH, 0, sizeof(unsigned));
        cudaMemset(d_trH, 0, sizeof(unsigned));
        cudaMemset(d_tsR, 0, tSegs * sizeof(unsigned));
        cudaMemset(d_nP, 0, sizeof(unsigned));

        launchKernel();
        printf(".");
    };

    // ── Validation run: full pipeline (kernel + L2L + L2P) ──
    resetAll();
    launchKernel();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        cudaGetLastError();
        goto cleanup;
    }

    {
        // Read back interaction counts
        cudaMemcpy(&res.m2lCount, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.p2pCount, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

        // Read back queue diagnostics
        cudaMemcpy(&res.iactWHead, d_wH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.iactRHead, d_rH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.travWHead, d_twH, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res.travRHead, d_trH, sizeof(unsigned), cudaMemcpyDeviceToHost);

        std::vector<unsigned> h_iSegReady(gSegs);
        std::vector<unsigned> h_iSegCount(gSegs);
        std::vector<unsigned> h_tSegReady(tSegs);
        cudaMemcpy(h_iSegReady.data(), d_sR, gSegs * sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_iSegCount.data(), d_segCount, gSegs * sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tSegReady.data(), d_tsR, tSegs * sizeof(unsigned), cudaMemcpyDeviceToHost);

        res.iactSegReadyBusy =
            static_cast<unsigned>(std::count_if(h_iSegReady.begin(), h_iSegReady.end(), [](unsigned v) { return v != 0u; }));
        res.iactSegCountBusy =
            static_cast<unsigned>(std::count_if(h_iSegCount.begin(), h_iSegCount.end(), [](unsigned v) { return v != 0u; }));
        res.travSegReadyBusy =
            static_cast<unsigned>(std::count_if(h_tSegReady.begin(), h_tSegReady.end(), [](unsigned v) { return v != 0u; }));

        // Check counts vs CPU reference
        res.countsMismatch = (res.m2lCount != refM2l || res.p2pCount != refP2p);

        // L2L downsweep + L2P to compute energy
        fmm::downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals, tables);
        checkGpuErrors(cudaDeviceSynchronize());

        TreeNodeIndex numLeafRange = lastLeafIndex - firstLeafIndex;
        if (numLeafRange > 0)
        {
            constexpr int numThreads = 256;
            fmm::l2pKernel<<<(numLeafRange + numThreads - 1) / numThreads, numThreads>>>(
                firstLeafIndex, lastLeafIndex, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z,
                d_ppot, d_pax, d_pay, d_paz, firstTarget, tables);
            checkGpuErrors(cudaDeviceSynchronize());
        }

        // Download and compute energy
        std::vector<T> h_ppot(numTargets);
        cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(T), cudaMemcpyDeviceToHost);

        LocalIndex lastTarget = firstTarget + numTargets;
        T ugravLoc = 0;
        for (LocalIndex t = firstTarget; t < lastTarget; ++t)
        {
            LocalIndex ti = t - firstTarget;
            ugravLoc += G * refMasses[t] * h_ppot[ti];
        }
        res.energy = T(0.5) * ugravLoc;
        res.energyError = std::abs(refEnergy) > 0 ? std::abs(res.energy - refEnergy) / std::abs(refEnergy) : 0;
    }

    // ── Warmup (kernel only) ──
    for (unsigned i = 0; i < nWarm; ++i)
        timeGpu(resetAndLaunchKernel);

    // ── Timed runs (kernel only) ──
    {
        std::vector<float> times(nRuns);
        for (unsigned i = 0; i < nRuns; ++i)
            times[i] = timeGpu(resetAndLaunchKernel);
        res.stats = computeStats(times);
        res.valid = true;
    }

cleanup:
    cudaFree(d_gA);
    cudaFree(d_gB);
    cudaFree(d_gIsP2P);
    cudaFree(d_wH);
    cudaFree(d_rH);
    cudaFree(d_segCount);
    cudaFree(d_sR);
    cudaFree(d_nP);
    cudaFree(d_tA);
    cudaFree(d_tB);
    cudaFree(d_twH);
    cudaFree(d_trH);
    cudaFree(d_tsR);
    cudaFree(d_m2lCount);
    cudaFree(d_p2pCount);
    return res;
}

// ── Config dispatch: benchmark + record one config for one numWarps ──────────

template<int NW, size_t CI, class T>
void benchAndRecord(
    std::vector<FmmTuneResult>& out,
    TreeNodeIndex* d_childOffsets,
    ryoanji::Vec3<T>* d_geoCenters,
    ryoanji::Vec3<T>* d_geoSizes,
    ryoanji::Vec4<T>* d_centers,
    fmm::SphericalMultipole<T>* d_multipoles,
    fmm::SphericalLocalExpansion<T>* d_locals,
    TreeNodeIndex* d_internalToLeaf,
    TreeNodeIndex* d_leafToInternal,
    ryoanji::LocalIndex* d_layout,
    T* d_x, T* d_y, T* d_z, T* d_h, T* d_m,
    T* d_ppot, T* d_pax, T* d_pay, T* d_paz,
    ryoanji::LocalIndex firstTarget,
    ryoanji::LocalIndex numTargets,
    TreeNodeIndex numNodes,
    TreeNodeIndex firstLeafIndex,
    TreeNodeIndex lastLeafIndex,
    fmm::GpuSphericalTables tables,
    std::span<const TreeNodeIndex> levelRange,
    unsigned refM2l, unsigned refP2p, double refEnergy,
    const T* refMasses, float G,
    unsigned nWarm, unsigned nRuns)
{
    constexpr auto c = tuneConfigs[CI];
    printf("  [nW=%d] Config %2zu: iCS=%-3u iFP=%-4u iAP=%-4u iAPo=%-3u iFPo=%-3u "
           "tCS=%-3u tFP=%-4u tAP=%-4u tAPo=%-3u ... ",
           NW, CI, c.chunkSize, c.forcePush, c.attemptPush, c.attemptPop, c.forcePop,
           c.travChunkSize, c.travForcePush, c.travAttemptPush, c.travAttemptPop);
    fflush(stdout);

    auto r = benchOneFmm<NW, FmmTuneTravConfig<CI, NW>, T>(
        c,
        d_childOffsets, d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals,
        d_internalToLeaf, d_leafToInternal, d_layout,
        d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz,
        firstTarget, numTargets, numNodes, firstLeafIndex, lastLeafIndex,
        tables, levelRange,
        refM2l, refP2p, refEnergy, refMasses, G,
        nWarm, nRuns);

    if (r.valid)
        printf("median=%.3f ms  Eerr=%.1e%s\n", r.stats.median, r.energyError,
               r.countsMismatch ? " COUNTS_MISMATCH!" : "");
    else
        printf("SKIPPED\n");

    out.push_back(r);
}

// ── Iterate over all configs for one numWarps value ──────────────────────────

template<int NW, class T, size_t... CIs>
void benchAllConfigs(
    std::index_sequence<CIs...>,
    std::vector<FmmTuneResult>& out,
    TreeNodeIndex* d_childOffsets,
    ryoanji::Vec3<T>* d_geoCenters,
    ryoanji::Vec3<T>* d_geoSizes,
    ryoanji::Vec4<T>* d_centers,
    fmm::SphericalMultipole<T>* d_multipoles,
    fmm::SphericalLocalExpansion<T>* d_locals,
    TreeNodeIndex* d_internalToLeaf,
    TreeNodeIndex* d_leafToInternal,
    ryoanji::LocalIndex* d_layout,
    T* d_x, T* d_y, T* d_z, T* d_h, T* d_m,
    T* d_ppot, T* d_pax, T* d_pay, T* d_paz,
    ryoanji::LocalIndex firstTarget,
    ryoanji::LocalIndex numTargets,
    TreeNodeIndex numNodes,
    TreeNodeIndex firstLeafIndex,
    TreeNodeIndex lastLeafIndex,
    fmm::GpuSphericalTables tables,
    std::span<const TreeNodeIndex> levelRange,
    unsigned refM2l, unsigned refP2p, double refEnergy,
    const T* refMasses, float G,
    unsigned nWarm, unsigned nRuns)
{
    (benchAndRecord<NW, CIs, T>(
         out,
         d_childOffsets, d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals,
         d_internalToLeaf, d_leafToInternal, d_layout,
         d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz,
         firstTarget, numTargets, numNodes, firstLeafIndex, lastLeafIndex,
         tables, levelRange,
         refM2l, refP2p, refEnergy, refMasses, G,
         nWarm, nRuns),
     ...);
}

// ── Iterate over all numWarps values ─────────────────────────────────────────

template<class T, int First, int... Rest>
void dispatchWarps(
    std::integer_sequence<int, First, Rest...>,
    std::vector<FmmTuneResult>& out,
    TreeNodeIndex* d_childOffsets,
    ryoanji::Vec3<T>* d_geoCenters,
    ryoanji::Vec3<T>* d_geoSizes,
    ryoanji::Vec4<T>* d_centers,
    fmm::SphericalMultipole<T>* d_multipoles,
    fmm::SphericalLocalExpansion<T>* d_locals,
    TreeNodeIndex* d_internalToLeaf,
    TreeNodeIndex* d_leafToInternal,
    ryoanji::LocalIndex* d_layout,
    T* d_x, T* d_y, T* d_z, T* d_h, T* d_m,
    T* d_ppot, T* d_pax, T* d_pay, T* d_paz,
    ryoanji::LocalIndex firstTarget,
    ryoanji::LocalIndex numTargets,
    TreeNodeIndex numNodes,
    TreeNodeIndex firstLeafIndex,
    TreeNodeIndex lastLeafIndex,
    fmm::GpuSphericalTables tables,
    std::span<const TreeNodeIndex> levelRange,
    unsigned refM2l, unsigned refP2p, double refEnergy,
    const T* refMasses, float G,
    unsigned nWarm, unsigned nRuns)
{
    printf("\n── numWarps = %d ──────────────────────────────────────────────────────────\n", First);
    benchAllConfigs<First, T>(
        std::make_index_sequence<numTuneConfigs>{},
        out,
        d_childOffsets, d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals,
        d_internalToLeaf, d_leafToInternal, d_layout,
        d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz,
        firstTarget, numTargets, numNodes, firstLeafIndex, lastLeafIndex,
        tables, levelRange,
        refM2l, refP2p, refEnergy, refMasses, G,
        nWarm, nRuns);

    if constexpr (sizeof...(Rest) > 0)
    {
        dispatchWarps<T>(
            std::integer_sequence<int, Rest...>{},
            out,
            d_childOffsets, d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals,
            d_internalToLeaf, d_leafToInternal, d_layout,
            d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz,
            firstTarget, numTargets, numNodes, firstLeafIndex, lastLeafIndex,
            tables, levelRange,
            refM2l, refP2p, refEnergy, refMasses, G,
            nWarm, nRuns);
    }
}

// ── P99/P90 error computation ─────────────────────────────────────────────────

template<class T>
static double computeP99(LocalIndex numParticles, const T* ax, const T* ay, const T* az, const T* refAx,
                         const T* refAy, const T* refAz)
{
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> aApprox{ax[i], ay[i], az[i]};
        ryoanji::Vec3<T> aRef{refAx[i], refAy[i], refAz[i]};
        T                refNorm = norm2(aRef);
        delta[i]                 = refNorm > 0 ? std::sqrt(norm2(aApprox - aRef) / refNorm) : 0;
    }
    std::sort(delta.begin(), delta.end());
    return double(delta[LocalIndex(numParticles * 0.99)]);
}

template<class T>
static double computeP90(LocalIndex numParticles, const T* ax, const T* ay, const T* az, const T* refAx,
                         const T* refAy, const T* refAz)
{
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> aApprox{ax[i], ay[i], az[i]};
        ryoanji::Vec3<T> aRef{refAx[i], refAy[i], refAz[i]};
        T                refNorm = norm2(aRef);
        delta[i]                 = refNorm > 0 ? std::sqrt(norm2(aApprox - aRef) / refNorm) : 0;
    }
    std::sort(delta.begin(), delta.end());
    return double(delta[LocalIndex(numParticles * 0.90)]);
}

// ── Main benchmark ────────────────────────────────────────────────────────────

void tuneFmmBenchmark(unsigned numParticles = 200000,
                       unsigned bucketSize   = 64,
                       unsigned numWarmup    = 3,
                       unsigned numRuns      = 10)
{
    using Tc            = double;
    using Tm            = Tc;
    using T             = Tc;
    using KeyType       = uint64_t;
    using BhMpole       = ryoanji::CartesianQuadrupole<T>;

    float          theta    = 0.5;
    float          G        = 1.0;
    T              invTheta = T(1) / T(theta);
    cstone::Box<T> box(-1, 1);

    // ── Open result file ──
    std::time_t now = std::time(nullptr);
    std::tm     tmNow{};
    localtime_r(&now, &tmNow);
    char resultFileName[128];
    std::strftime(resultFileName, sizeof(resultFileName), "fmm-tune-%Y%m%d-%H%M%S.txt", &tmNow);

    char resultFilePath[256];
    std::snprintf(resultFilePath, sizeof(resultFilePath), "fmm/test/tuning/%s", resultFileName);
    mkdir("fmm", 0755);
    mkdir("fmm/test", 0755);
    mkdir("fmm/test/tuning", 0755);
    FILE* resultFile = std::fopen(resultFilePath, "w");
    if (!resultFile)
    {
        std::snprintf(resultFilePath, sizeof(resultFilePath), "%s", resultFileName);
        resultFile = std::fopen(resultFilePath, "w");
    }
    if (!resultFile) { std::perror("Could not open result file"); }

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 1: Generate particles, GPU tree build, source centers + MAC
    // ════════════════════════════════════════════════════════════════════════

    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("  FMM Dual Traversal Parameter Tuning Benchmark\n");
    printf("════════════════════════════════════════════════════════════════════════════════\n");

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);
    coordinates.adjustH(2, 5);

    std::vector<T> masses(numParticles, T(1) / numParticles);

    // Upload to GPU
    thrust::device_vector<T> d_x(coordinates.x().begin(), coordinates.x().end());
    thrust::device_vector<T> d_y(coordinates.y().begin(), coordinates.y().end());
    thrust::device_vector<T> d_z(coordinates.z().begin(), coordinates.z().end());
    thrust::device_vector<T> d_m(masses.begin(), masses.end());
    thrust::device_vector<T> d_h(coordinates.h().begin(), coordinates.h().end());

    // GPU octree build
    ryoanji::TreeBuilder<KeyType> treeBuilder(bucketSize);
    int numSources = treeBuilder.update(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), numParticles, box);
    // x,y,z now SFC-sorted on device; h,m stay aligned (input was already SFC-sorted)

    unsigned              highestLevel = treeBuilder.maxTreeLevel();
    const TreeNodeIndex*  levelRange   = treeBuilder.levelRange();
    TreeNodeIndex         numLeaves    = treeBuilder.numLeafNodes();
    TreeNodeIndex         numNodes     = TreeNodeIndex(numSources);
    std::span<const TreeNodeIndex> levelRangeSpan(levelRange, highestLevel + 2);

    // GPU source centers + MAC
    thrust::device_vector<SourceCenterType<T>> d_centers(numSources);
    cstone::computeLeafSourceCenterGpu(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                       treeBuilder.leafToInternal(), numLeaves,
                                       treeBuilder.layout(), rawPtr(d_centers));
    cstone::upsweepCentersGpu(highestLevel, levelRange,
                              treeBuilder.childOffsets(), rawPtr(d_centers));
    cstone::setMacGpu(treeBuilder.nodeKeys(), numNodes, rawPtr(d_centers), float(invTheta), box);

    TreeNodeIndex firstLeafIdx = 0;
    TreeNodeIndex lastLeafIdx  = numLeaves;
    LocalIndex    firstTarget  = 0;
    LocalIndex    numTargets   = numParticles;

    printf("  particles=%u  leaves=%d  nodes=%d  bucket=%u  P=%d\n",
           numParticles, numLeaves, numNodes, bucketSize, fmm::ExpansionOrder);
    printf("  warmup=%u  runs=%u  bpc=%u  configs=%zu  numWarps values=%zu\n",
           numWarmup, numRuns, kBpc, numTuneConfigs, TuneWarpList::size());
    printf("════════════════════════════════════════════════════════════════════════════════\n");

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 2: BH GPU reference (replaces CPU FMM reference)
    // ════════════════════════════════════════════════════════════════════════

    printf("\nRunning BH GPU reference...\n");

    // BH multipoles
    thrust::device_vector<BhMpole> d_bhMultipoles(numSources);
    thrust::fill(d_bhMultipoles.begin(), d_bhMultipoles.end(), BhMpole{});

    ryoanji::computeLeafMultipoles(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                   treeBuilder.leafToInternal(), numLeaves, treeBuilder.layout(),
                                   rawPtr(d_centers), rawPtr(d_bhMultipoles));

    for (int level = int(highestLevel) - 1; level >= 1; --level)
    {
        TreeNodeIndex first = levelRange[level];
        TreeNodeIndex last  = levelRange[level + 1];
        if (first < last)
        {
            ryoanji::upsweepMultipoles(first, last, treeBuilder.childOffsets(), rawPtr(d_centers),
                                       rawPtr(d_bhMultipoles));
        }
    }

    // Read root childOffset
    TreeNodeIndex rootChildOffset;
    checkGpuErrors(
        cudaMemcpy(&rootChildOffset, treeBuilder.childOffsets(), sizeof(TreeNodeIndex), cudaMemcpyDeviceToHost));

    // BH traversal
    thrust::device_vector<T> d_bhAx(numParticles, 0), d_bhAy(numParticles, 0), d_bhAz(numParticles, 0);

    cstone::GroupData<cstone::GpuTag> groups;
    cstone::computeFixedGroups(LocalIndex(0), numParticles, ryoanji::bhMaxTargetSize(), groups);
    thrust::device_vector<int> globalPool(ryoanji::stackSize(groups.numGroups));

    ryoanji::traverse(groups.view(), rootChildOffset,
                      rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                      rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                      treeBuilder.childOffsets(), treeBuilder.internalToLeaf(), treeBuilder.layout(),
                      rawPtr(d_centers), rawPtr(d_bhMultipoles), T(G), 0,
                      ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
                      (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                      thrust::raw_pointer_cast(globalPool.data()));
    checkGpuErrors(cudaDeviceSynchronize());

    // Download BH accelerations for accuracy comparison
    std::vector<T> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
    thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
    thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
    thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

    printf("  BH GPU reference computed.\n");

    // Free BH-only device memory
    d_bhMultipoles.clear();
    d_bhMultipoles.shrink_to_fit();
    d_bhAx.clear(); d_bhAx.shrink_to_fit();
    d_bhAy.clear(); d_bhAy.shrink_to_fit();
    d_bhAz.clear(); d_bhAz.shrink_to_fit();

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 3: GPU FMM setup — geo centers, multipoles, locals, accumulators
    // ════════════════════════════════════════════════════════════════════════

    printf("\nSetting up FMM on GPU...\n");

    // Upload spherical tables
    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    // Compute geometric centers/sizes on GPU
    thrust::device_vector<Vec3<T>> d_geoCenters(numNodes);
    thrust::device_vector<Vec3<T>> d_geoSizes(numNodes);
    cstone::computeGeoCentersGpu(treeBuilder.nodeKeys(), numNodes, rawPtr(d_geoCenters), rawPtr(d_geoSizes), box);

    // Scale geo sizes by invTheta for scalar MAC
    {
        int nt = 256;
        int nb = cstone::iceil(numNodes, nt);
        if (nb) { fmm::sphScaleVec3Kernel<<<nb, nt>>>(rawPtr(d_geoSizes), numNodes, T(invTheta)); }
    }

    // Compute FMM multipoles on GPU
    thrust::device_vector<fmm::SphericalMultipole<T>> d_multipoles(numNodes);
    checkGpuErrors(cudaMemset(rawPtr(d_multipoles), 0, numNodes * sizeof(fmm::SphericalMultipole<T>)));

    fmm::computeLeafMultipolesGpu(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                   treeBuilder.leafToInternal(), numLeaves, treeBuilder.layout(),
                                   rawPtr(d_centers), rawPtr(d_multipoles), tables);
    fmm::upsweepMultipolesGpu(levelRangeSpan, treeBuilder.childOffsets(), rawPtr(d_centers),
                               rawPtr(d_multipoles), tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // Allocate locals and particle accumulators (re-zeroed per config run)
    thrust::device_vector<fmm::SphericalLocalExpansion<T>> d_locals(numNodes);
    thrust::device_vector<T> d_ppot(numTargets), d_pax(numTargets), d_pay(numTargets), d_paz(numTargets);

    printf("  GPU FMM setup complete. numNodes=%d numTargets=%d\n", numNodes, numTargets);

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 4: Sweep all configs (pass refM2l=0, refP2p=0, refEnergy=0)
    // ════════════════════════════════════════════════════════════════════════

    std::vector<FmmTuneResult> results;
    results.reserve(numTuneConfigs * TuneWarpList::size());

    printf("\nRunning benchmarks...\n");
    dispatchWarps<T>(
        TuneWarpList{}, results,
        const_cast<TreeNodeIndex*>(treeBuilder.childOffsets()),
        rawPtr(d_geoCenters), rawPtr(d_geoSizes),
        reinterpret_cast<Vec4<T>*>(rawPtr(d_centers)),
        rawPtr(d_multipoles),
        rawPtr(d_locals),
        const_cast<TreeNodeIndex*>(treeBuilder.internalToLeaf()),
        const_cast<TreeNodeIndex*>(treeBuilder.leafToInternal()),
        const_cast<LocalIndex*>(treeBuilder.layout()),
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m),
        rawPtr(d_ppot), rawPtr(d_pax), rawPtr(d_pay), rawPtr(d_paz),
        firstTarget, numTargets, numNodes, firstLeafIdx, lastLeafIdx,
        tables, levelRangeSpan,
        0u, 0u, 0.0, masses.data(), G,
        numWarmup, numRuns);

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 4b: Post-process — fix up counts/energy using first valid config
    // ════════════════════════════════════════════════════════════════════════

    unsigned refM2l    = 0;
    unsigned refP2p    = 0;
    double   refEnergy = 0;
    for (const auto& r : results)
    {
        if (r.valid)
        {
            refM2l    = r.m2lCount;
            refP2p    = r.p2pCount;
            refEnergy = r.energy;
            break;
        }
    }

    // Recompute countsMismatch and energyError relative to first valid config
    for (auto& r : results)
    {
        if (!r.valid) continue;
        r.countsMismatch = (r.m2lCount != refM2l || r.p2pCount != refP2p);
        r.energyError    = std::abs(refEnergy) > 0 ? std::abs(r.energy - refEnergy) / std::abs(refEnergy) : 0;
    }

    // Compute FMM p90/p99 vs BH (run the GPU-native spherical FMM)
    double fmmP90 = 0, fmmP99 = 0;
    {
        thrust::device_vector<T> d_fmmAx(numParticles, 0), d_fmmAy(numParticles, 0), d_fmmAz(numParticles, 0);
        T fmmEnergy = 0;

        fmm::computeGravityFMMGpu(
            treeBuilder.nodeKeys(), treeBuilder.childOffsets(),
            treeBuilder.internalToLeaf(), treeBuilder.leafToInternal(),
            treeBuilder.layout(), rawPtr(d_centers),
            levelRangeSpan, numNodes, numLeaves,
            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m),
            box, G, 1.0f / theta, rawPtr(d_fmmAx), rawPtr(d_fmmAy), rawPtr(d_fmmAz),
            &fmmEnergy, numParticles);

        std::vector<T> fmmAx(numParticles), fmmAy(numParticles), fmmAz(numParticles);
        thrust::copy(d_fmmAx.begin(), d_fmmAx.end(), fmmAx.begin());
        thrust::copy(d_fmmAy.begin(), d_fmmAy.end(), fmmAy.begin());
        thrust::copy(d_fmmAz.begin(), d_fmmAz.end(), fmmAz.begin());

        fmmP90 = computeP90(LocalIndex(numParticles), fmmAx.data(), fmmAy.data(), fmmAz.data(),
                            bhAx.data(), bhAy.data(), bhAz.data());
        fmmP99 = computeP99(LocalIndex(numParticles), fmmAx.data(), fmmAy.data(), fmmAz.data(),
                            bhAx.data(), bhAy.data(), bhAz.data());
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 5: Report
    // ════════════════════════════════════════════════════════════════════════

    // Find production baseline (nW=7, production FmmTravConfig params)
    float baselineMedian = 0.f;
    for (const auto& r : results)
    {
        if (r.valid && r.numWarps == 7 &&
            r.params.chunkSize == 224 && r.params.forcePush == 704 &&
            r.params.attemptPush == 512 && r.params.attemptPop == 160 &&
            r.params.forcePop == 96 &&
            r.params.travChunkSize == 192 && r.params.travForcePush == 640 &&
            r.params.travAttemptPush == 224 && r.params.travAttemptPop == 32)
        {
            baselineMedian = r.stats.median;
            break;
        }
    }
    if (baselineMedian == 0.f)
    {
        for (const auto& r : results)
        {
            if (r.valid) { baselineMedian = r.stats.median; break; }
        }
    }

    printf("\n");
    printDual(resultFile, "════════════════════════════════════════════════════════════════════════════════"
                          "════════════════════════════════════════════════════════════════════════════════\n");
    printDual(resultFile, "  FMM Dual Traversal Parameter Tuning Results\n");
    printDual(resultFile, "  particles=%u  leaves=%d  nodes=%d  bucket=%u  P=%d\n",
              numParticles, numLeaves, numNodes, bucketSize, fmm::ExpansionOrder);
    printDual(resultFile, "  1st config ref: M2L=%u  P2P=%u  energy=%.10e\n", refM2l, refP2p, refEnergy);
    printDual(resultFile, "  FMM vs BH: p90=%.2e  p99=%.2e\n", fmmP90, fmmP99);
    printDual(resultFile, "  baseline (nW=7 prod): %.3f ms\n\n", baselineMedian);

    printDual(resultFile,
              "  nW | iCS  iFP  iAP iAPo iFPo | tCS  tFP  tAP tAPo |"
              "  median    mean  stddev | m2l       p2p       |"
              " iW/iR       tW/tR       | iRdy iCnt tRdy | Eerr       | speedup\n");
    printDual(resultFile,
              "  ---+-------------------------+---------------------+"
              "------------------------+---------------------+"
              "-------------------------+------------------+------------+---------\n");

    float bestMedian = 1e30f;
    int   bestIdx    = -1;

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (!r.valid)
        {
            printDual(resultFile,
                      "  %2d | %3u %4u %4u %3u %4u | %3u %4u %4u %3u |"
                      " %-22s | --        --        |"
                      " --                      | --               | --         | --\n",
                      r.numWarps,
                      r.params.chunkSize, r.params.forcePush, r.params.attemptPush,
                      r.params.attemptPop, r.params.forcePop,
                      r.params.travChunkSize, r.params.travForcePush,
                      r.params.travAttemptPush, r.params.travAttemptPop,
                      "SKIPPED");
            continue;
        }

        const char* note = r.countsMismatch ? " MISMATCH!" : "";
        float speedup = (baselineMedian > 0) ? baselineMedian / r.stats.median : 0.f;

        printDual(resultFile,
                  "  %2d | %3u %4u %4u %3u %4u | %3u %4u %4u %3u |"
                  " %7.3f %7.3f %7.3f | %-9u %-9u |"
                  " %6u/%-6u %6u/%-6u | %4u %4u %4u | %.3e | %5.2fx%s\n",
                  r.numWarps,
                  r.params.chunkSize, r.params.forcePush, r.params.attemptPush,
                  r.params.attemptPop, r.params.forcePop,
                  r.params.travChunkSize, r.params.travForcePush,
                  r.params.travAttemptPush, r.params.travAttemptPop,
                  r.stats.median, r.stats.mean, r.stats.stddev,
                  r.m2lCount, r.p2pCount,
                  r.iactWHead, r.iactRHead, r.travWHead, r.travRHead,
                  r.iactSegReadyBusy, r.iactSegCountBusy, r.travSegReadyBusy,
                  r.energyError, speedup, note);

        if (r.stats.median < bestMedian)
        {
            bestMedian = r.stats.median;
            bestIdx    = int(i);
        }
    }

    printDual(resultFile,
              "  ---+-------------------------+---------------------+"
              "------------------------+---------------------+"
              "-------------------------+------------------+------------+---------\n");

    if (bestIdx >= 0)
    {
        const auto& b = results[bestIdx];
        float bestSpeedup = (baselineMedian > 0) ? baselineMedian / b.stats.median : 0.f;
        printDual(resultFile,
                  "\n  BEST:  nW=%d  iCS=%u iFP=%u iAP=%u iAPo=%u iFPo=%u  "
                  "tCS=%u tFP=%u tAP=%u tAPo=%u  =>  median=%.3f ms  (%.2fx vs production)\n",
                  b.numWarps,
                  b.params.chunkSize, b.params.forcePush, b.params.attemptPush,
                  b.params.attemptPop, b.params.forcePop,
                  b.params.travChunkSize, b.params.travForcePush,
                  b.params.travAttemptPush, b.params.travAttemptPop,
                  b.stats.median, bestSpeedup);
    }

    // Top 25 fastest
    std::vector<FmmTuneResult> fastest;
    fastest.reserve(results.size());
    for (const auto& r : results)
    {
        if (r.valid) fastest.push_back(r);
    }
    std::sort(fastest.begin(), fastest.end(),
              [](const FmmTuneResult& a, const FmmTuneResult& b) { return a.stats.median < b.stats.median; });

    size_t topN = std::min<size_t>(25, fastest.size());
    printDual(resultFile, "\nTop %zu Fastest Runs:\n", topN);
    printDual(resultFile, "  rk | nW | iCS  iFP  iAP iAPo iFPo | tCS  tFP  tAP tAPo | median    mean  stddev | Eerr       | speedup\n");
    printDual(resultFile, "  ---+----+-------------------------+---------------------+------------------------+------------+---------\n");
    for (size_t i = 0; i < topN; ++i)
    {
        const auto& r = fastest[i];
        float speedup = (baselineMedian > 0) ? baselineMedian / r.stats.median : 0.f;
        printDual(resultFile,
                  "  %2zu | %2d | %3u %4u %4u %3u %4u | %3u %4u %4u %3u | %7.3f %7.3f %7.3f | %.3e | %6.2fx\n",
                  i + 1, r.numWarps,
                  r.params.chunkSize, r.params.forcePush, r.params.attemptPush,
                  r.params.attemptPop, r.params.forcePop,
                  r.params.travChunkSize, r.params.travForcePush,
                  r.params.travAttemptPush, r.params.travAttemptPop,
                  r.stats.median, r.stats.mean, r.stats.stddev,
                  r.energyError, speedup);
    }

    printDual(resultFile, "\nResult file: %s\n", resultFilePath);
    printf("════════════════════════════════════════════════════════════════════════════════\n");

    if (resultFile) std::fclose(resultFile);

    // ── Cleanup ──
    fmm::freeSphericalTables(tables);
}

} // anonymous namespace

TEST(SphericalFMM, tuneFmmBenchmark) { tuneFmmBenchmark(200000, 64, 3, 10); }
