/*
 * Cartesian quadrupole FMM GPU accuracy test vs direct sum (FCC lattice)
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU accuracy test: Cartesian quadrupole FMM vs direct sum on FCC lattice
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Validates FMM (dual traversal) accuracy against direct sum for pure point masses
 * on a structured FCC lattice. Unlike random distributions, FCC particles have no
 * SPH smoothing — a tiny h (1e-10) is used solely to avoid the self-interaction
 * singularity in P2P.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "coord_samples/face_centered_cubic.hpp"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

TEST(CartesianQpoleFMM, FccGpuAccuracyVsDirectSum)
{
    using T       = double;
    using KeyType = uint64_t;
    using FmmMpole = fmm::CartesianMultipole<T>;

    float          theta      = 0.5f;
    float          G          = 1.0f;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(0, 1);

    FaceCenteredCubicCoordinates<T, SfcKind<KeyType>> coordinates(16, 16, 16, box);
    LocalIndex numParticles = coordinates.x().size(); // 4 * 16^3 = 16384

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    // Tiny softening to avoid self-interaction singularity (h=0 → rsqrt(0) → NaN)
    std::vector<T> h(numParticles, T(1e-10));
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
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal,
                                   layout.data(), centers.data());
    upsweep(octree.levelRange, octree.childOffsets.data(), centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    float invTheta = 1.0f / theta;

    // ===== Direct sum reference =====
    std::vector<T> refAx(numParticles, 0), refAy(numParticles, 0), refAz(numParticles, 0);
    std::vector<T> refPot(numParticles, 0);
    ryoanji::directSum(x, y, z, h.data(), masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, 0,
                       refAx.data(), refAy.data(), refAz.data(), refPot.data());

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
        octree.numLeafNodes, x, y, z, h.data(), masses.data(), box, G, invTheta, (T*)nullptr, fmmAx.data(),
        fmmAy.data(), fmmAz.data(), &fmmEgrav, numParticles);

    auto fmmDelta = computeErrors(fmmAx, fmmAy, fmmAz);

    printf("FMM GPU accuracy on FCC lattice (N=%d, theta=%.1f):\n", numParticles, theta);
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

    // Assertions
    float fmmP99 = float(fmmDelta[LocalIndex(numParticles * 0.99)]);
    float fmmMax = float(fmmDelta[numParticles - 1]);

    EXPECT_LT(fmmP99, 2e-1) << "FMM p99 error too large";
    EXPECT_LT(fmmMax, 1.0) << "FMM max error too large";
}
