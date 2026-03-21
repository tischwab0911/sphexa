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
 * Measures GPU traversal time for FMM (DirectionalMac)
 * and Barnes-Hut across a range of particle counts to find the crossover point
 * where FMM's O(N) scaling overtakes BH's O(N log N).
 *
 * At small N (<=100k), also runs a direct sum for accuracy validation.
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
#include "ryoanji/nbody/kernel.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

struct ScalingResult
{
    unsigned N;
    float    fmmDir;     // GPU traversal ms (total FMM phases)
    float    bhMs;       // GPU BH traversal ms
};

static ScalingResult runScalingPoint(unsigned N, float theta, unsigned bucketSize, const cstone::Box<double>& box,
                                     float G)
{
    using T       = double;
    using KeyType = uint64_t;
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

    // Compute FMM multipoles
    std::vector<FmmMpole> fmmMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   fmmMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), fmmMultipoles.data());

    float invTheta = 1.0f / theta;

    // --- Run FMM with DirectionalMac ---
    auto runFmm = [&](auto macTag) -> fmm::FmmGpuStats
    {
        std::vector<T> ax(numParticles, 0), ay(numParticles, 0), az(numParticles, 0);
        T              egrav = 0;
        fmm::FmmGpuStats stats;

        fmm::computeGravityFMMGpu<decltype(macTag)::value>(
            octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
            std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
            octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, ax.data(), ay.data(),
            az.data(), &egrav, numParticles, &stats);

        return stats;
    };

    auto dirStats = runFmm(std::integral_constant<fmm::MacVariant, fmm::DirectionalMac>{});

    // --- Run GPU Barnes-Hut with CUDA event timing ---
    std::vector<BhMpole> bhMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   bhMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), bhMultipoles.data());

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

    // Warmup BH
    ryoanji::traverse(
        groups.view(), octree.childOffsets[0],
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout),
        rawPtr(d_centers), rawPtr(d_bhMultipoles),
        T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
        (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
        thrust::raw_pointer_cast(globalPool.data()));

    // Reset BH output
    thrust::fill(d_bhAx.begin(), d_bhAx.end(), T(0));
    thrust::fill(d_bhAy.begin(), d_bhAy.end(), T(0));
    thrust::fill(d_bhAz.begin(), d_bhAz.end(), T(0));

    // Timed BH run
    cudaEvent_t bhStart, bhEnd;
    checkGpuErrors(cudaEventCreate(&bhStart));
    checkGpuErrors(cudaEventCreate(&bhEnd));

    checkGpuErrors(cudaEventRecord(bhStart));
    ryoanji::traverse(
        groups.view(), octree.childOffsets[0],
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout),
        rawPtr(d_centers), rawPtr(d_bhMultipoles),
        T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
        (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
        thrust::raw_pointer_cast(globalPool.data()));
    checkGpuErrors(cudaEventRecord(bhEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    float bhMs = 0;
    checkGpuErrors(cudaEventElapsedTime(&bhMs, bhStart, bhEnd));
    checkGpuErrors(cudaEventDestroy(bhStart));
    checkGpuErrors(cudaEventDestroy(bhEnd));

    // --- Accuracy validation at small N ---
    if (N <= 100000)
    {
        // Direct sum reference
        std::vector<T> refAx(numParticles, 0), refAy(numParticles, 0), refAz(numParticles, 0);
        std::vector<T> refPot(numParticles, 0);
        ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, 0,
                           refAx.data(), refAy.data(), refAz.data(), refPot.data());

        // FMM DirectionalMac accuracy check
        std::vector<T> fmmDirAx(numParticles, 0), fmmDirAy(numParticles, 0), fmmDirAz(numParticles, 0);
        T              fmmDirEgrav = 0;
        fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
            octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
            std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
            octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, fmmDirAx.data(),
            fmmDirAy.data(), fmmDirAz.data(), &fmmDirEgrav, numParticles);

        // BH accuracy check
        std::vector<T> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
        thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
        thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
        thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

        auto computeP99 = [&](const std::vector<T>& ax, const std::vector<T>& ay, const std::vector<T>& az,
                               const char* label)
        {
            std::vector<T> delta(numParticles);
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                ryoanji::Vec3<T> aApprox{ax[i], ay[i], az[i]};
                ryoanji::Vec3<T> aRef{refAx[i], refAy[i], refAz[i]};
                T refNorm = norm2(aRef);
                delta[i]  = refNorm > 0 ? std::sqrt(norm2(aApprox - aRef) / refNorm) : 0;
            }
            std::sort(delta.begin(), delta.end());
            T p99 = delta[LocalIndex(numParticles * 0.99)];
            printf("  %s p99 error: %.6f\n", label, p99);
        };

        printf("  Accuracy at N=%u:\n", N);
        computeP99(fmmDirAx, fmmDirAy, fmmDirAz, "FMM Dir");
        computeP99(bhAx, bhAy, bhAz, "BH     ");
    }

    return {N, dirStats.msTotal(), bhMs};
}

TEST(CartesianQpoleFMM, ScalingBenchmark)
{
    float          theta      = 0.5f;
    float          G          = 1.0f;
    unsigned       bucketSize = 64;
    cstone::Box<double> box(-1, 1);

    std::vector<unsigned> sizes = {50000, 100000, 200000, 500000, 1000000, 2000000, 5000000};
    std::vector<ScalingResult> results;

    for (unsigned N : sizes)
    {
        printf("\n--- Running N = %u ---\n", N);
        auto result = runScalingPoint(N, theta, bucketSize, box, G);
        results.push_back(result);
    }

    // Print summary table
    printf("\n=== FMM vs BH Scaling (theta=%.1f, bucket=%u) ===\n", theta, bucketSize);
    printf("       N | FMM Dir  |    BH\n");
    printf("---------|----------|--------\n");
    for (const auto& r : results)
    {
        printf(" %7u | %6.2f ms | %5.2f ms\n", r.N, r.fmmDir, r.bhMs);
    }
    printf("\n");
}
