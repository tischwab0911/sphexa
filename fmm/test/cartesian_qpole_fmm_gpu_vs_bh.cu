/*
 * Cartesian quadrupole FMM GPU vs GPU Barnes-Hut benchmark
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Benchmark GPU Cartesian quadrupole FMM vs GPU Barnes-Hut across particle counts
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * For each particle count N:
 *   - Builds octree, computes multipoles on CPU, uploads to GPU
 *   - Runs FMM (DirectionalMac) with warmup + timed runs, collecting per-phase CUDA event timing
 *   - Runs BH with warmup + timed runs using CUDA event timing
 *   - At small N (<=100k), validates accuracy against direct sum reference
 *   - Reports median timings across runs
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
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/groups_gpu.h"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

namespace
{

struct BenchmarkResult
{
    unsigned N;

    // FMM median timings (ms)
    float fmmTotal, fmmUpsweep, fmmTraversal, fmmL2L, fmmL2P;

    // BH median timing (ms)
    float bhMs;

    // Accuracy (only valid for N <= 100k)
    bool  hasAccuracy;
    float fmmP99Err, bhP99Err;
};

static float median(std::vector<float>& v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static BenchmarkResult runBenchmarkPoint(unsigned N, float theta, unsigned bucketSize, const cstone::Box<double>& box,
                                          float G, int nWarmup, int nRuns)
{
    using T        = double;
    using KeyType  = uint64_t;
    using FmmMpole = fmm::CartesianMultipole<T>;
    using BhMpole  = ryoanji::CartesianQuadrupole<T>;

    LocalIndex numParticles = N;

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);
    coordinates.adjustH(2, 5);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();
    const T* h = coordinates.h().data();

    std::vector<T> masses(numParticles, T(1) / numParticles);

    // Build octree
    auto [treeLeaves, counts] = computeOctree(std::span(coordinates.particleKeys()), bucketSize);

    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    // Compute centers of mass + MAC
    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets.data(), centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    float invTheta = 1.0f / theta;

    // Compute FMM multipoles
    std::vector<FmmMpole> fmmMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   fmmMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), fmmMultipoles.data());

    // ===== FMM benchmark: warmup + timed runs =====
    // computeGravityFMMGpu allocates/frees GPU memory per call, but internal CUDA event timing
    // (via FmmGpuStats) measures only kernel execution, excluding alloc/dealloc overhead.

    auto runFmmOnce = [&]() -> fmm::FmmGpuStats
    {
        std::vector<T> ax(numParticles, 0), ay(numParticles, 0), az(numParticles, 0);
        T              egrav = 0;
        fmm::FmmGpuStats stats;

        fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
            octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
            std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
            octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, ax.data(), ay.data(),
            az.data(), &egrav, numParticles, &stats);

        return stats;
    };

    // Warmup
    for (int i = 0; i < nWarmup; ++i)
        runFmmOnce();

    // Timed runs
    std::vector<float> fmmTotals, fmmUpsweeps, fmmTraversals, fmmL2Ls, fmmL2Ps;
    for (int i = 0; i < nRuns; ++i)
    {
        auto stats = runFmmOnce();
        fmmTotals.push_back(stats.msTotal());
        fmmUpsweeps.push_back(stats.msUpsweep);
        fmmTraversals.push_back(stats.msTraversal);
        fmmL2Ls.push_back(stats.msL2L);
        fmmL2Ps.push_back(stats.msL2P);
    }

    // ===== BH benchmark: upload once, warmup + timed runs =====
    std::vector<BhMpole> bhMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   bhMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), bhMultipoles.data());

    // Upload BH data to device (persists across runs)
    thrust::device_vector<T>                   d_x(x, x + numParticles), d_y(y, y + numParticles),
                                               d_z(z, z + numParticles), d_h(h, h + numParticles);
    thrust::device_vector<T>                   d_m(masses);
    thrust::device_vector<TreeNodeIndex>       d_childOffsets(octree.childOffsets);
    thrust::device_vector<TreeNodeIndex>       d_internalToLeaf(octree.internalToLeaf);
    thrust::device_vector<LocalIndex>          d_layout(layout);
    thrust::device_vector<SourceCenterType<T>> d_centers(centers);
    thrust::device_vector<BhMpole>             d_bhMultipoles(bhMultipoles);

    thrust::device_vector<T> d_bhAx(numParticles, 0), d_bhAy(numParticles, 0), d_bhAz(numParticles, 0);

    cstone::GroupData<cstone::GpuTag> groups;
    cstone::computeFixedGroups(LocalIndex(0), numParticles, ryoanji::bhMaxTargetSize(), groups);
    thrust::device_vector<int> globalPool(ryoanji::stackSize(groups.numGroups));

    auto runBhOnce = [&]()
    {
        thrust::fill(d_bhAx.begin(), d_bhAx.end(), T(0));
        thrust::fill(d_bhAy.begin(), d_bhAy.end(), T(0));
        thrust::fill(d_bhAz.begin(), d_bhAz.end(), T(0));

        ryoanji::traverse(groups.view(), octree.childOffsets[0], rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                          rawPtr(d_h), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout), rawPtr(d_centers),
                          rawPtr(d_bhMultipoles), T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
                          (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                          thrust::raw_pointer_cast(globalPool.data()));
    };

    // BH warmup
    for (int i = 0; i < nWarmup; ++i)
        runBhOnce();

    // BH timed runs
    cudaEvent_t bhStart, bhEnd;
    checkGpuErrors(cudaEventCreate(&bhStart));
    checkGpuErrors(cudaEventCreate(&bhEnd));

    std::vector<float> bhTimes;
    for (int i = 0; i < nRuns; ++i)
    {
        thrust::fill(d_bhAx.begin(), d_bhAx.end(), T(0));
        thrust::fill(d_bhAy.begin(), d_bhAy.end(), T(0));
        thrust::fill(d_bhAz.begin(), d_bhAz.end(), T(0));

        checkGpuErrors(cudaEventRecord(bhStart));
        ryoanji::traverse(groups.view(), octree.childOffsets[0], rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                          rawPtr(d_h), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                          rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout), rawPtr(d_centers),
                          rawPtr(d_bhMultipoles), T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
                          (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                          thrust::raw_pointer_cast(globalPool.data()));
        checkGpuErrors(cudaEventRecord(bhEnd));
        checkGpuErrors(cudaDeviceSynchronize());

        float ms = 0;
        checkGpuErrors(cudaEventElapsedTime(&ms, bhStart, bhEnd));
        bhTimes.push_back(ms);
    }

    checkGpuErrors(cudaEventDestroy(bhStart));
    checkGpuErrors(cudaEventDestroy(bhEnd));

    BenchmarkResult result{};
    result.N            = N;
    result.fmmTotal     = median(fmmTotals);
    result.fmmUpsweep   = median(fmmUpsweeps);
    result.fmmTraversal = median(fmmTraversals);
    result.fmmL2L       = median(fmmL2Ls);
    result.fmmL2P       = median(fmmL2Ps);
    result.bhMs         = median(bhTimes);
    result.hasAccuracy  = false;
    result.fmmP99Err    = 0;
    result.bhP99Err     = 0;

    // ===== Accuracy validation at small N =====
    if (N <= 100000)
    {
        // Direct sum reference
        std::vector<T> refAx(numParticles, 0), refAy(numParticles, 0), refAz(numParticles, 0);
        std::vector<T> refPot(numParticles, 0);
        ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, 0,
                           refAx.data(), refAy.data(), refAz.data(), refPot.data());

        auto computeP99 = [&](const std::vector<T>& ax, const std::vector<T>& ay, const std::vector<T>& az) -> float
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
            return float(delta[LocalIndex(numParticles * 0.99)]);
        };

        // FMM accuracy run
        std::vector<T> fmmAx(numParticles, 0), fmmAy(numParticles, 0), fmmAz(numParticles, 0);
        T              fmmEgrav = 0;
        fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
            octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
            std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
            octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, fmmAx.data(), fmmAy.data(),
            fmmAz.data(), &fmmEgrav, numParticles);

        result.fmmP99Err = computeP99(fmmAx, fmmAy, fmmAz);

        // BH accuracy (copy last BH result from device)
        std::vector<T> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
        thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
        thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
        thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

        result.bhP99Err   = computeP99(bhAx, bhAy, bhAz);
        result.hasAccuracy = true;

        EXPECT_LT(result.fmmP99Err, 2e-1) << "FMM p99 error too large at N=" << N;
        EXPECT_LT(result.bhP99Err, 2e-1) << "BH p99 error too large at N=" << N;
    }

    return result;
}

} // namespace

TEST(CartesianQpoleFMM, BenchmarkFMMvsBH)
{
    float               theta      = 0.5f;
    float               G          = 1.0f;
    unsigned            bucketSize = 64;
    cstone::Box<double> box(-1, 1);

    constexpr int nWarmup = 3;
    constexpr int nRuns   = 5;

    std::vector<unsigned>        sizes = {50000, 100000, 200000, 500000, 1000000};
    std::vector<BenchmarkResult> results;

    for (unsigned N : sizes)
    {
        printf("\n--- N = %u (%d warmup + %d timed runs) ---\n", N, nWarmup, nRuns);
        results.push_back(runBenchmarkPoint(N, theta, bucketSize, box, G, nWarmup, nRuns));
    }

    // Print summary table
    printf("\n=== FMM vs BH Benchmark (theta=%.1f, bucket=%u, %d warmup + %d runs, median) ===\n", theta, bucketSize,
           nWarmup, nRuns);
    printf("       N |  FMM total | FMM trav | FMM ups | FMM L2L | FMM L2P |       BH | speedup | FMM err  | BH err\n");
    printf("---------|------------|----------|---------|---------|---------|----------|---------|----------|---------\n");

    for (const auto& r : results)
    {
        float speedup = (r.bhMs > 0) ? r.bhMs / r.fmmTotal : 0;

        if (r.hasAccuracy)
        {
            printf(" %7u | %8.2f ms | %6.2f ms | %5.2f ms | %5.2f ms | %5.2f ms | %6.2f ms | %5.2fx | %.2e | %.2e\n",
                   r.N, r.fmmTotal, r.fmmTraversal, r.fmmUpsweep, r.fmmL2L, r.fmmL2P, r.bhMs, speedup, r.fmmP99Err,
                   r.bhP99Err);
        }
        else
        {
            printf(" %7u | %8.2f ms | %6.2f ms | %5.2f ms | %5.2f ms | %5.2f ms | %6.2f ms | %5.2fx |    --    |    --\n",
                   r.N, r.fmmTotal, r.fmmTraversal, r.fmmUpsweep, r.fmmL2L, r.fmmL2P, r.bhMs, speedup);
        }
    }
    printf("\n");
}
