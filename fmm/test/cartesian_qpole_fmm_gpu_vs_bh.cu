/*
 * Cartesian quadrupole FMM & BH GPU accuracy test vs direct sum
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU accuracy test: Cartesian quadrupole FMM and BH vs direct sum
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Validates that both FMM (dual traversal) and Barnes-Hut on GPU produce
 * correct accelerations compared to a direct sum reference.
 * This is a correctness test, not a benchmark.
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

TEST(CartesianQpoleFMM, GpuAccuracyVsDirectSum)
{
    using T       = double;
    using KeyType = uint64_t;
    using FmmMpole = fmm::CartesianMultipole<T>;
    using BhMpole  = ryoanji::CartesianQuadrupole<T>;

    float          theta      = 0.5f;
    float          G          = 1.0f;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1);
    LocalIndex     numParticles = 10000;

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

    // ===== Direct sum reference =====
    std::vector<T> refAx(numParticles, 0), refAy(numParticles, 0), refAz(numParticles, 0);
    std::vector<T> refPot(numParticles, 0);
    ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, 0, refAx.data(),
                       refAy.data(), refAz.data(), refPot.data());

    auto computeErrors = [&](const std::vector<T>& ax, const std::vector<T>& ay, const std::vector<T>& az)
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
        return delta;
    };

    // ===== FMM accuracy =====
    std::vector<FmmMpole> fmmMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   fmmMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), fmmMultipoles.data());

    std::vector<T> fmmAx(numParticles, 0), fmmAy(numParticles, 0), fmmAz(numParticles, 0);
    T              fmmEgrav = 0;
    fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
        octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
        std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
        octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, fmmAx.data(), fmmAy.data(),
        fmmAz.data(), &fmmEgrav, numParticles);

    auto fmmDelta = computeErrors(fmmAx, fmmAy, fmmAz);

    printf("FMM GPU accuracy (N=%d, theta=%.1f):\n", numParticles, theta);
    printf("  min=%.2e  50th=%.2e  90th=%.2e  99th=%.2e  max=%.2e\n", fmmDelta[0],
           fmmDelta[numParticles / 2], fmmDelta[LocalIndex(numParticles * 0.9)],
           fmmDelta[LocalIndex(numParticles * 0.99)], fmmDelta[numParticles - 1]);

    // Energy error
    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
        refPotSum += refPot[i];
    refPotSum *= 0.5;
    double fmmEnergyRelErr = std::abs(fmmEgrav - refPotSum) / std::abs(refPotSum);
    printf("  Energy relative error: %.2e\n", fmmEnergyRelErr);

    // ===== BH accuracy =====
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

    ryoanji::traverse(groups.view(), octree.childOffsets[0], rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                      rawPtr(d_h), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                      rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout), rawPtr(d_centers),
                      rawPtr(d_bhMultipoles), T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()}, (T*)nullptr,
                      rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
                      thrust::raw_pointer_cast(globalPool.data()));

    std::vector<T> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
    thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
    thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
    thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

    auto bhDelta = computeErrors(bhAx, bhAy, bhAz);

    printf("BH GPU accuracy (N=%d, theta=%.1f):\n", numParticles, theta);
    printf("  min=%.2e  50th=%.2e  90th=%.2e  99th=%.2e  max=%.2e\n", bhDelta[0],
           bhDelta[numParticles / 2], bhDelta[LocalIndex(numParticles * 0.9)],
           bhDelta[LocalIndex(numParticles * 0.99)], bhDelta[numParticles - 1]);

    // Assertions
    float fmmP99 = float(fmmDelta[LocalIndex(numParticles * 0.99)]);
    float fmmMax = float(fmmDelta[numParticles - 1]);
    float bhP99  = float(bhDelta[LocalIndex(numParticles * 0.99)]);
    float bhMax  = float(bhDelta[numParticles - 1]);

    EXPECT_LT(fmmP99, 2e-1) << "FMM p99 error too large";
    EXPECT_LT(fmmMax, 1.0) << "FMM max error too large";
    EXPECT_LT(bhP99, 2e-1) << "BH p99 error too large";
    EXPECT_LT(bhMax, 1.0) << "BH max error too large";
}
