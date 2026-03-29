/*
 * FMM vs BH scaling benchmark
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Scaling benchmark: GPU Cartesian quadrupole FMM (DirectionalMac) vs GPU Barnes-Hut
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Measures GPU traversal time for FMM (DirectionalMac) and Barnes-Hut across
 * a range of particle counts to find the crossover point where FMM's O(N)
 * scaling overtakes BH's O(N log N).
 *
 * Uses GPU octree build (TreeBuilder) and GPU-native FMM overloads to keep
 * all data on device. Only particle generation and p99 computation touch the CPU.
 *
 * Uses p99 relative acceleration error (vs direct sum at small N, vs spherical
 * FMM at large N) instead of energy error for accuracy validation.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/groups_gpu.h"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/treebuilder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/kernel.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/upsweep_gpu.h"

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include "fmm/cartesian_qpole_fmm.cuh"
#include "fmm/spherical_multipole.cuh"

using namespace cstone;

namespace
{

struct BenchmarkResult
{
    unsigned N;
    float    fmmMedian, fmmMean, fmmStdev;
    float    fmmUpsweepMed, fmmTraversalMed, fmmL2LMed, fmmL2PMed;
    float    bhMedian, bhMean, bhStdev;
    bool     hasAccuracy;
    double   fmmP99Err, bhP99Err;
    double   sphP99Err;
};

static float benchMedian(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static float benchMean(const std::vector<float>& v)
{
    return std::accumulate(v.begin(), v.end(), 0.0f) / float(v.size());
}

static float benchStdev(const std::vector<float>& v, float m)
{
    float sumSq = 0;
    for (float x : v)
        sumSq += (x - m) * (x - m);
    return std::sqrt(sumSq / float(v.size() - 1));
}

template<class T, class Tref = T>
static double computeP99(LocalIndex numParticles, const T* ax, const T* ay, const T* az, const Tref* refAx,
                         const Tref* refAy, const Tref* refAz)
{
    std::vector<double> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        double dx = double(ax[i]) - double(refAx[i]);
        double dy = double(ay[i]) - double(refAy[i]);
        double dz = double(az[i]) - double(refAz[i]);
        double refNorm = double(refAx[i]) * double(refAx[i]) + double(refAy[i]) * double(refAy[i]) +
                         double(refAz[i]) * double(refAz[i]);
        double deltaNorm = dx * dx + dy * dy + dz * dz;
        delta[i]         = refNorm > 0 ? std::sqrt(deltaNorm / refNorm) : 0;
    }
    std::sort(delta.begin(), delta.end());
    return delta[LocalIndex(numParticles * 0.99)];
}

static BenchmarkResult runBenchmarkPoint(unsigned N, float theta, unsigned bucketSize, const cstone::Box<double>& box,
                                         float G, int nWarmup, int nRuns)
{
    using Tc      = double;
    using Tm      = double;
    using KeyType = uint64_t;
    using BhMpole = ryoanji::CartesianQuadrupole<Tm>;

    LocalIndex numParticles = N;
    float      invTheta     = 1.0f / theta;

    // ===== Particle generation (CPU → GPU, one-time upload) =====
    RandomGaussianCoordinates<Tc, SfcKind<KeyType>> coordinates(numParticles, box);

    thrust::device_vector<Tc> d_x(coordinates.x().begin(), coordinates.x().end());
    thrust::device_vector<Tc> d_y(coordinates.y().begin(), coordinates.y().end());
    thrust::device_vector<Tc> d_z(coordinates.z().begin(), coordinates.z().end());
    thrust::device_vector<Tm> d_m(numParticles, Tm(1) / numParticles);
    thrust::device_vector<Tm> d_h(numParticles, Tm(0.01));

    // ===== GPU octree build =====
    ryoanji::TreeBuilder<KeyType> treeBuilder(bucketSize);
    int numSources = treeBuilder.update(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), numParticles, box);
    // x,y,z now SFC-sorted on device

    unsigned              highestLevel = treeBuilder.maxTreeLevel();
    const TreeNodeIndex*  levelRange   = treeBuilder.levelRange();
    TreeNodeIndex         numLeaves    = treeBuilder.numLeafNodes();
    std::span<const TreeNodeIndex> levelRangeSpan(levelRange, highestLevel + 2);

    // ===== GPU source centers + MAC =====
    thrust::device_vector<SourceCenterType<Tc>> d_centers(numSources);
    cstone::computeLeafSourceCenterGpu(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                       treeBuilder.leafToInternal(), numLeaves,
                                       treeBuilder.layout(), rawPtr(d_centers));
    cstone::upsweepCentersGpu(highestLevel, levelRange,
                              treeBuilder.childOffsets(), rawPtr(d_centers));
    cstone::setMacGpu(treeBuilder.nodeKeys(), TreeNodeIndex(numSources), rawPtr(d_centers), invTheta, box);

    // ===== FMM benchmark: warmup + timed runs =====
    struct FmmRunResult
    {
        fmm::FmmGpuStats stats;
        Tc               egrav;
    };

    thrust::device_vector<Tm> d_fmmAx(numParticles), d_fmmAy(numParticles), d_fmmAz(numParticles);

    auto runFmmOnce = [&]() -> FmmRunResult
    {
        checkGpuErrors(cudaMemset(rawPtr(d_fmmAx), 0, numParticles * sizeof(Tm)));
        checkGpuErrors(cudaMemset(rawPtr(d_fmmAy), 0, numParticles * sizeof(Tm)));
        checkGpuErrors(cudaMemset(rawPtr(d_fmmAz), 0, numParticles * sizeof(Tm)));
        Tc               egrav = 0;
        fmm::FmmGpuStats stats;

        fmm::computeGravityFMMGpu<fmm::DirectionalMac, Tc, KeyType, Tm>(
            treeBuilder.nodeKeys(), treeBuilder.childOffsets(),
            treeBuilder.internalToLeaf(), treeBuilder.leafToInternal(),
            treeBuilder.layout(), rawPtr(d_centers),
            levelRangeSpan, TreeNodeIndex(numSources), numLeaves,
            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m),
            box, G, invTheta, rawPtr(d_fmmAx), rawPtr(d_fmmAy), rawPtr(d_fmmAz),
            &egrav, numParticles, &stats);

        return {stats, egrav};
    };

    for (int i = 0; i < nWarmup; ++i)
        runFmmOnce();

    // Profiled FMM run (captured by nsys/ncu with --capture-range=cudaProfilerApi)
    {
        char nvtxLabel[64];
        snprintf(nvtxLabel, sizeof(nvtxLabel), "FMM N=%u", N);
        cudaProfilerStart();
        nvtxRangePush(nvtxLabel);
        runFmmOnce();
        nvtxRangePop();
        cudaProfilerStop();
    }

    std::vector<float> fmmTotals, fmmUpsweeps, fmmTraversals, fmmL2Ls, fmmL2Ps;
    for (int i = 0; i < nRuns; ++i)
    {
        auto [stats, egrav] = runFmmOnce();
        fmmTotals.push_back(stats.msTotal());
        fmmUpsweeps.push_back(stats.msUpsweep);
        fmmTraversals.push_back(stats.msTraversal);
        fmmL2Ls.push_back(stats.msL2L);
        fmmL2Ps.push_back(stats.msL2P);
    }

    // Extra FMM run to capture accelerations for p99 (only when accuracy is computed)
    if (N <= 5000000) { runFmmOnce(); }

    // ===== BH benchmark =====
    // BH multipoles — allocated on device, recomputed each timed run
    thrust::device_vector<BhMpole> d_bhMultipoles(numSources);

    // Read root childOffset from device (single scalar download)
    TreeNodeIndex rootChildOffset;
    checkGpuErrors(
        cudaMemcpy(&rootChildOffset, treeBuilder.childOffsets(), sizeof(TreeNodeIndex), cudaMemcpyDeviceToHost));

    thrust::device_vector<Tm> d_bhAx(numParticles, 0), d_bhAy(numParticles, 0), d_bhAz(numParticles, 0);

    cstone::GroupData<cstone::GpuTag> groups;
    cstone::computeFixedGroups(LocalIndex(0), numParticles, ryoanji::bhMaxTargetSize(), groups);
    thrust::device_vector<int> globalPool(ryoanji::stackSize(groups.numGroups));

    auto bhUpsweep = [&]()
    {
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
    };

    auto runBhOnce = [&]()
    {
        bhUpsweep();

        thrust::fill(d_bhAx.begin(), d_bhAx.end(), Tm(0));
        thrust::fill(d_bhAy.begin(), d_bhAy.end(), Tm(0));
        thrust::fill(d_bhAz.begin(), d_bhAz.end(), Tm(0));

        ryoanji::traverse(groups.view(), rootChildOffset,
                          rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          treeBuilder.childOffsets(), treeBuilder.internalToLeaf(), treeBuilder.layout(),
                          rawPtr(d_centers), rawPtr(d_bhMultipoles), Tc(G), 0,
                          ryoanji::Vec3<Tc>{box.lx(), box.ly(), box.lz()},
                          (Tm*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                          thrust::raw_pointer_cast(globalPool.data()));
    };

    // BH warmup
    for (int i = 0; i < nWarmup; ++i)
        runBhOnce();

    // Profiled BH run (captured by nsys/ncu with --capture-range=cudaProfilerApi)
    {
        char nvtxLabel[64];
        snprintf(nvtxLabel, sizeof(nvtxLabel), "BH N=%u", N);
        cudaProfilerStart();
        nvtxRangePush(nvtxLabel);
        runBhOnce();
        nvtxRangePop();
        cudaProfilerStop();
    }

    // BH timed runs
    cudaEvent_t bhStart, bhEnd;
    checkGpuErrors(cudaEventCreate(&bhStart));
    checkGpuErrors(cudaEventCreate(&bhEnd));

    std::vector<float> bhTimes;
    for (int i = 0; i < nRuns; ++i)
    {
        thrust::fill(d_bhMultipoles.begin(), d_bhMultipoles.end(), BhMpole{});

        checkGpuErrors(cudaEventRecord(bhStart));

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

        thrust::fill(d_bhAx.begin(), d_bhAx.end(), Tm(0));
        thrust::fill(d_bhAy.begin(), d_bhAy.end(), Tm(0));
        thrust::fill(d_bhAz.begin(), d_bhAz.end(), Tm(0));

        ryoanji::traverse(groups.view(), rootChildOffset,
                          rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          treeBuilder.childOffsets(), treeBuilder.internalToLeaf(), treeBuilder.layout(),
                          rawPtr(d_centers), rawPtr(d_bhMultipoles), Tc(G), 0,
                          ryoanji::Vec3<Tc>{box.lx(), box.ly(), box.lz()},
                          (Tm*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                          thrust::raw_pointer_cast(globalPool.data()));

        checkGpuErrors(cudaEventRecord(bhEnd));
        checkGpuErrors(cudaDeviceSynchronize());

        float ms = 0;
        checkGpuErrors(cudaEventElapsedTime(&ms, bhStart, bhEnd));
        bhTimes.push_back(ms);
    }

    checkGpuErrors(cudaEventDestroy(bhStart));
    checkGpuErrors(cudaEventDestroy(bhEnd));

    // ===== Spherical FMM for reference (N <= 5M) =====
    // Spherical FMM uses single-type T interface, so we need Tc-precision h/m
    thrust::device_vector<Tc> d_sphAx, d_sphAy, d_sphAz;
    thrust::device_vector<Tc> d_refH, d_refM;
    if (N <= 5000000)
    {
        d_sphAx.resize(numParticles, 0);
        d_sphAy.resize(numParticles, 0);
        d_sphAz.resize(numParticles, 0);
        d_refH.assign(numParticles, Tc(0.01));
        d_refM.assign(numParticles, Tc(1) / numParticles);
        Tc sphEnergy = 0;

        fmm::computeGravityFMMGpu(
            treeBuilder.nodeKeys(), treeBuilder.childOffsets(),
            treeBuilder.internalToLeaf(), treeBuilder.leafToInternal(),
            treeBuilder.layout(), rawPtr(d_centers),
            levelRangeSpan, TreeNodeIndex(numSources), numLeaves,
            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_refH), rawPtr(d_refM),
            box, G, 1.0f / theta, rawPtr(d_sphAx), rawPtr(d_sphAy), rawPtr(d_sphAz),
            &sphEnergy, numParticles);
    }

    // ===== Compute statistics =====
    BenchmarkResult result{};
    result.N = N;

    float fmmM        = benchMean(fmmTotals);
    result.fmmMedian   = benchMedian(fmmTotals);
    result.fmmMean     = fmmM;
    result.fmmStdev    = benchStdev(fmmTotals, fmmM);

    result.fmmUpsweepMed   = benchMedian(fmmUpsweeps);
    result.fmmTraversalMed = benchMedian(fmmTraversals);
    result.fmmL2LMed       = benchMedian(fmmL2Ls);
    result.fmmL2PMed       = benchMedian(fmmL2Ps);

    float bhM         = benchMean(bhTimes);
    result.bhMedian    = benchMedian(bhTimes);
    result.bhMean      = bhM;
    result.bhStdev     = benchStdev(bhTimes, bhM);

    result.sphP99Err   = 0;
    result.hasAccuracy = false;
    result.fmmP99Err   = 0;
    result.bhP99Err    = 0;

    // ===== P99 acceleration error computation (only for N <= 5M) =====
    if (N <= 5000000)
    {
        // Download FMM, BH, spherical accelerations to CPU
        std::vector<Tm> fmmAx(numParticles), fmmAy(numParticles), fmmAz(numParticles);
        thrust::copy(d_fmmAx.begin(), d_fmmAx.end(), fmmAx.begin());
        thrust::copy(d_fmmAy.begin(), d_fmmAy.end(), fmmAy.begin());
        thrust::copy(d_fmmAz.begin(), d_fmmAz.end(), fmmAz.begin());

        std::vector<Tm> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
        thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
        thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
        thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

        std::vector<Tc> sphAx(numParticles), sphAy(numParticles), sphAz(numParticles);
        thrust::copy(d_sphAx.begin(), d_sphAx.end(), sphAx.begin());
        thrust::copy(d_sphAy.begin(), d_sphAy.end(), sphAy.begin());
        thrust::copy(d_sphAz.begin(), d_sphAz.end(), sphAz.begin());

        if (N <= 100000)
        {
            // Small N: GPU direct sum as reference (at Tc precision)
            thrust::device_vector<Tc> d_refAx(numParticles, 0), d_refAy(numParticles, 0),
                                      d_refAz(numParticles, 0), d_refPot(numParticles, 0);
            ryoanji::directSum(size_t(0), size_t(numParticles), size_t(numParticles),
                               ryoanji::Vec3<Tc>{box.lx(), box.ly(), box.lz()}, 0,
                               rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_refM), rawPtr(d_refH),
                               rawPtr(d_refPot), rawPtr(d_refAx), rawPtr(d_refAy), rawPtr(d_refAz));

            // Download and apply G-scaling (GPU directSum doesn't include G)
            std::vector<Tc> refAx(numParticles), refAy(numParticles), refAz(numParticles);
            thrust::copy(d_refAx.begin(), d_refAx.end(), refAx.begin());
            thrust::copy(d_refAy.begin(), d_refAy.end(), refAy.begin());
            thrust::copy(d_refAz.begin(), d_refAz.end(), refAz.begin());
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                refAx[i] *= G;
                refAy[i] *= G;
                refAz[i] *= G;
            }

            result.fmmP99Err = computeP99(numParticles, fmmAx.data(), fmmAy.data(), fmmAz.data(), refAx.data(),
                                          refAy.data(), refAz.data());
            result.bhP99Err  = computeP99(numParticles, bhAx.data(), bhAy.data(), bhAz.data(), refAx.data(),
                                          refAy.data(), refAz.data());
            result.sphP99Err = computeP99(numParticles, sphAx.data(), sphAy.data(), sphAz.data(), refAx.data(),
                                          refAy.data(), refAz.data());
            result.hasAccuracy = true;

            EXPECT_LT(result.fmmP99Err, 1e-1) << "Cartesian FMM p99 error too large at N=" << N;
            EXPECT_LT(result.bhP99Err, 1e-1) << "BH p99 error too large at N=" << N;
            EXPECT_LT(result.sphP99Err, 1e-2) << "Spherical FMM p99 error too large at N=" << N;
        }
        else
        {
            // Medium N: spherical FMM as reference
            result.fmmP99Err = computeP99(numParticles, fmmAx.data(), fmmAy.data(), fmmAz.data(), sphAx.data(),
                                          sphAy.data(), sphAz.data());
            result.bhP99Err  = computeP99(numParticles, bhAx.data(), bhAy.data(), bhAz.data(), sphAx.data(),
                                          sphAy.data(), sphAz.data());
            result.sphP99Err   = 0;
            result.hasAccuracy = true;

            EXPECT_LT(result.fmmP99Err, 1e-1) << "Cartesian FMM p99 error vs spherical too large at N=" << N;
            EXPECT_LT(result.bhP99Err, 1e-1) << "BH p99 error vs spherical too large at N=" << N;
        }
    }
    // N > 5M: no accuracy computation, just timing

    return result;
}

} // namespace

TEST(CartesianFMM, BenchmarkFMMvsBH)
{
    float               theta      = 0.5f;
    float               G          = 1.0f;
    unsigned            bucketSize = 64;
    cstone::Box<double> box(-1, 1);

    std::vector<unsigned> sizes = {50000,     100000,    200000,    500000,     1000000,   2000000,  5000000,
                                   10000000,  20000000,  50000000,  100000000,  200000000, 400000000};
    std::vector<BenchmarkResult> results;

    for (unsigned N : sizes)
    {
        int nWarmup = (N <= 5000000) ? 5 : 3;
        int nRuns   = (N <= 5000000) ? 25 : 10;
        auto r = runBenchmarkPoint(N, theta, bucketSize, box, G, nWarmup, nRuns);
        results.push_back(r);

        // Print per-N result immediately
        float speedup = (r.bhMedian > 0) ? r.bhMedian / r.fmmMedian : 0;
        char  fmmP99Str[16] = "    --   ";
        char  bhP99Str[16]  = "    --   ";
        char  sphP99Str[16] = "    --   ";
        if (r.hasAccuracy)
        {
            snprintf(fmmP99Str, sizeof(fmmP99Str), "%.2e", r.fmmP99Err);
            snprintf(bhP99Str, sizeof(bhP99Str), "%.2e", r.bhP99Err);
            if (r.N <= 100000) { snprintf(sphP99Str, sizeof(sphP99Str), "%.2e", r.sphP99Err); }
            else { snprintf(sphP99Str, sizeof(sphP99Str), "  (ref)  "); }
        }
        printf("[N=%9u] FMM %6.2f ms  BH %6.2f ms  speedup %5.2fx  FMM-p99 %s  BH-p99 %s  Sph-p99 %s\n", r.N,
               r.fmmMedian, r.bhMedian, speedup, fmmP99Str, bhP99Str, sphP99Str);
        fflush(stdout);
    }

    // Print summary table
    printf("\n=== FMM vs BH Benchmark (theta=%.1f, bucket=%u, 5+25 runs N<=5M, 3+10 runs N>5M) ===\n", theta,
           bucketSize);
    printf("       N | FMM med  | FMM mean |  BH med  |  BH mean | FMM/BH  | FMM p99  |  BH p99  | Sph p99\n");
    printf("---------|----------|----------|----------|----------|---------|----------|----------|---------\n");

    for (const auto& r : results)
    {
        float speedup = (r.bhMedian > 0) ? r.bhMedian / r.fmmMedian : 0;

        char fmmP99Str[16] = "    --   ";
        char bhP99Str[16]  = "    --   ";
        char sphP99Str[16] = "    --   ";

        if (r.hasAccuracy)
        {
            snprintf(fmmP99Str, sizeof(fmmP99Str), "%.2e", r.fmmP99Err);
            snprintf(bhP99Str, sizeof(bhP99Str), "%.2e", r.bhP99Err);
            if (r.N <= 100000) { snprintf(sphP99Str, sizeof(sphP99Str), "%.2e", r.sphP99Err); }
            else { snprintf(sphP99Str, sizeof(sphP99Str), "  (ref)  "); }
        }

        printf(" %7u | %5.2f ms | %5.2f ms | %5.2f ms | %5.2f ms | %5.2fx | %s | %s | %s\n", r.N, r.fmmMedian,
               r.fmmMean, r.bhMedian, r.bhMean, speedup, fmmP99Str, bhP99Str, sphP99Str);
    }

    // FMM phase breakdown
    printf("\nFMM phase breakdown (median, ms):\n");
    printf("       N | upsweep  | traversal |   L2L    |   L2P\n");
    printf("---------|----------|-----------|----------|--------\n");
    for (const auto& r : results)
    {
        printf(" %7u | %5.2f ms | %6.2f ms | %5.2f ms | %5.2f ms\n", r.N, r.fmmUpsweepMed, r.fmmTraversalMed,
               r.fmmL2LMed, r.fmmL2PMed);
    }
    printf("\n");
}
