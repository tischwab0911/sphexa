/*
 * Cartesian FMM profile benchmark
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Lightweight profile target for GPU Cartesian quadrupole FMM dual traversal
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Runs the production computeGravityFMMGpu path with a single configuration
 * for use with nsys/ncu profiling. Separated from tune_cartesian_fmm.cu to
 * avoid pulling in the full tuning sweep infrastructure.
 */

#include <vector>
#include <numeric>
#include <cstdio>

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/cartesian_qpole_fmm.hpp"
#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

TEST(CartesianFMM, ProfileDualTraversal)
{
    using T       = double;
    using KeyType = uint64_t;
    using Mpole   = fmm::CartesianMultipole<T>;

    unsigned            N          = 2000000;
    float               theta      = 0.5f;
    float               G          = 1.0f;
    unsigned            bucketSize = 64;
    cstone::Box<double> box(-1, 1);

    // Build octree + multipoles
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(N, box);
    coordinates.adjustH(2, 5);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();
    const T* h = coordinates.h().data();

    std::vector<T> masses(N, T(1) / N);

    auto [treeLeaves, counts] = computeOctree(std::span(coordinates.particleKeys()), bucketSize);

    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets.data(), centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<Mpole> multipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   multipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    float invTheta = 1.0f / theta;

    auto runOnce = [&](fmm::FmmGpuStats* stats) -> T
    {
        std::vector<T> ax(N, 0), ay(N, 0), az(N, 0);
        T              egrav = 0;

        fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
            octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
            std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), multipoles.data(), layout.data(), 0,
            octree.numLeafNodes, x, y, z, h, masses.data(), box, G, invTheta, (T*)nullptr, ax.data(), ay.data(),
            az.data(), &egrav, N, stats);

        return egrav;
    };

    // Warmup
    runOnce(nullptr);

    // Profiled run
    fmm::FmmGpuStats stats;
    T                 energy = runOnce(&stats);

    printf("ProfileDualTraversal: N=%u  energy=%.10e\n", N, energy);
    printf("  upsweep=%.2f ms  traversal=%.2f ms  L2L=%.2f ms  L2P=%.2f ms  total=%.2f ms\n", stats.msUpsweep,
           stats.msTraversal, stats.msL2L, stats.msL2P, stats.msTotal());
}
