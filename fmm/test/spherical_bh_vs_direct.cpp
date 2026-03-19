/*
 * Spherical harmonic Barnes-Hut test
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test spherical harmonic BH against direct sum
 *
 * Builds an octree, computes spherical harmonic multipoles (P=6),
 * runs a Barnes-Hut traversal, and compares against a direct sum reference.
 */

#include <chrono>
#include <numeric>

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/spherical_multipole.hpp"

using namespace cstone;

TEST(SphericalBH, VsDirectSum)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = fmm::SphericalMultipole<T>;

    float          theta      = 0.5;
    float          G          = 1.0;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1);
    int            numShells    = 0;
    LocalIndex     numParticles = 10000;

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);
    coordinates.adjustH(2, 5);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();
    const T* h = coordinates.h().data();

    std::vector<T> masses(numParticles, T(1) / numParticles);

    // build octree
    auto [treeLeaves, counts] = computeOctree(std::span(coordinates.particleKeys()), bucketSize);

    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    // compute centers of mass
    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets.data(), centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    // compute spherical multipoles (P2M + M2M upsweep)
    std::vector<MultipoleType> multipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    // Barnes-Hut tree walk
    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    auto t0       = std::chrono::high_resolution_clock::now();
    T    egravTot = 0;
    fmm::computeGravity(octree.childOffsets.data(), octree.parents.data(), octree.internalToLeaf.data(), centers.data(),
                         multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h, masses.data(), box, G,
                         (T*)nullptr, ax.data(), ay.data(), az.data(), &egravTot, numShells);
    auto   t1      = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Spherical BH (P=" << fmm::ExpansionOrder << ") for " << numParticles << " particles: " << elapsed
              << " s" << std::endl;

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, numShells, Ax.data(),
                       Ay.data(), Az.data(), potentialReference.data());
    t1      = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Direct sum: " << elapsed << " s" << std::endl;

    // check total gravitational energy
    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        refPotSum += potentialReference[i];
    }
    refPotSum *= 0.5;
    EXPECT_NEAR(std::abs(refPotSum - egravTot) / std::abs(refPotSum), 0, 1e-2);

    // relative acceleration errors
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> axi{ax[i], ay[i], az[i]}, Axi{Ax[i], Ay[i], Az[i]};
        delta[i] = std::sqrt(norm2(axi - Axi) / norm2(Axi));
    }

    std::sort(begin(delta), end(delta));

    std::cout.precision(10);
    std::cout << "min Error: " << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numParticles / 2] << std::endl;
    std::cout << "90th percentile: " << delta[LocalIndex(numParticles * 0.9)] << std::endl;
    std::cout << "99th percentile: " << delta[LocalIndex(numParticles * 0.99)] << std::endl;
    std::cout << "max Error: " << delta[numParticles - 1] << std::endl;

    EXPECT_LT(delta[LocalIndex(numParticles * 0.99)], 1e-3);
    EXPECT_LT(delta[numParticles - 1], 1e-2);
}
