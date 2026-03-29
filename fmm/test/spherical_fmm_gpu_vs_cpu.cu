/*
 * Spherical harmonic FMM GPU vs CPU test
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test GPU FMM against CPU FMM and direct sum reference
 *
 * Builds an octree, runs both CPU and GPU FMM (dual traversal), and
 * compares results. Also validates against CPU direct sum reference.
 */

#include <chrono>
#include <numeric>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/spherical_multipole.hpp"
#include "fmm/spherical_multipole.cuh"

using namespace cstone;

TEST(SphericalFMM, GpuVsCpu)
{
    using Tc            = double;
    using Tm            = Tc;
    using T             = Tc;
    using KeyType       = uint64_t;
    using MultipoleType = fmm::SphericalMultipole<T>;

    float          theta      = 0.5;
    float          G          = 1.0;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1);
    LocalIndex     numParticles = 50000;

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

    // compute spherical multipoles (P2M + M2M upsweep) on CPU
    std::vector<MultipoleType> multipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   multipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    // ----- CPU FMM -----
    std::vector<T> cpuAx(numParticles, 0);
    std::vector<T> cpuAy(numParticles, 0);
    std::vector<T> cpuAz(numParticles, 0);
    T              cpuEgrav = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    fmm::computeGravityFMM<fmm::ScalarMac, T, KeyType>(
                                        octree.prefixes.data(), octree.childOffsets.data(),
                                        octree.internalToLeaf.data(), toInternal,
                                        std::span<const TreeNodeIndex>(octree.levelRange), centers.data(),
                                        multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h,
                                        masses.data(), box, G, 1.0f / theta, (T*)nullptr, cpuAx.data(), cpuAy.data(),
                                        cpuAz.data(), &cpuEgrav);
    auto   t1         = std::chrono::high_resolution_clock::now();
    double cpuElapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "CPU FMM (P=" << fmm::ExpansionOrder << ") for " << numParticles << " particles: " << cpuElapsed
              << " s" << std::endl;

    // ----- GPU FMM -----
    std::vector<T> gpuAx(numParticles, 0);
    std::vector<T> gpuAy(numParticles, 0);
    std::vector<T> gpuAz(numParticles, 0);
    T              gpuEgrav = 0;

    t0 = std::chrono::high_resolution_clock::now();
    fmm::computeGravityFMMGpu<fmm::ScalarMac, T, KeyType>(
        octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
        std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), multipoles.data(), layout.data(), 0,
        octree.numLeafNodes, x, y, z, h, masses.data(), box, G, 1.0f / theta, (T*)nullptr, gpuAx.data(),
        gpuAy.data(), gpuAz.data(), &gpuEgrav, numParticles);
    t1                 = std::chrono::high_resolution_clock::now();
    double gpuElapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "GPU FMM (P=" << fmm::ExpansionOrder << ") for " << numParticles << " particles: " << gpuElapsed
              << " s" << std::endl;

    // ----- GPU vs CPU comparison -----
    std::cout << "CPU energy: " << cpuEgrav << "  GPU energy: " << gpuEgrav << std::endl;
    EXPECT_NEAR(std::abs(cpuEgrav - gpuEgrav) / std::abs(cpuEgrav), 0, 1e-4)
        << "Energy mismatch: CPU=" << cpuEgrav << " GPU=" << gpuEgrav;

    std::vector<T> deltaGpuVsCpu(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> aGpu{gpuAx[i], gpuAy[i], gpuAz[i]};
        ryoanji::Vec3<T> aCpu{cpuAx[i], cpuAy[i], cpuAz[i]};
        T                cpuNorm = norm2(aCpu);
        deltaGpuVsCpu[i] = cpuNorm > 0 ? std::sqrt(norm2(aGpu - aCpu) / cpuNorm) : 0;
    }

    std::sort(deltaGpuVsCpu.begin(), deltaGpuVsCpu.end());

    std::cout.precision(10);
    std::cout << "GPU vs CPU:" << std::endl;
    std::cout << "  50th percentile: " << deltaGpuVsCpu[numParticles / 2] << std::endl;
    std::cout << "  99th percentile: " << deltaGpuVsCpu[LocalIndex(numParticles * 0.99)] << std::endl;
    std::cout << "  max error:       " << deltaGpuVsCpu[numParticles - 1] << std::endl;

    EXPECT_LT(deltaGpuVsCpu[LocalIndex(numParticles * 0.99)], 1e-5);
    EXPECT_LT(deltaGpuVsCpu[numParticles - 1], 1e-3);

    // ----- Direct sum reference (CPU) -----
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, 0, Ax.data(),
                       Ay.data(), Az.data(), potentialReference.data());
    t1                  = std::chrono::high_resolution_clock::now();
    double directElapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Direct sum (CPU): " << directElapsed << " s" << std::endl;

    // GPU FMM vs direct sum
    std::vector<T> deltaGpuVsDirect(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> aGpu{gpuAx[i], gpuAy[i], gpuAz[i]};
        ryoanji::Vec3<T> aRef{Ax[i], Ay[i], Az[i]};
        T                refNorm = norm2(aRef);
        deltaGpuVsDirect[i] = refNorm > 0 ? std::sqrt(norm2(aGpu - aRef) / refNorm) : 0;
    }

    std::sort(deltaGpuVsDirect.begin(), deltaGpuVsDirect.end());

    std::cout << "GPU FMM vs Direct:" << std::endl;
    std::cout << "  50th percentile: " << deltaGpuVsDirect[numParticles / 2] << std::endl;
    std::cout << "  99th percentile: " << deltaGpuVsDirect[LocalIndex(numParticles * 0.99)] << std::endl;
    std::cout << "  max error:       " << deltaGpuVsDirect[numParticles - 1] << std::endl;

    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
        refPotSum += potentialReference[i];
    refPotSum *= 0.5;

    std::cout << "Direct energy: " << refPotSum << "  GPU FMM energy: " << gpuEgrav << std::endl;
    EXPECT_NEAR(std::abs(refPotSum - gpuEgrav) / std::abs(refPotSum), 0, 1e-2);

    EXPECT_LT(deltaGpuVsDirect[LocalIndex(numParticles * 0.99)], 1e-3);
    EXPECT_LT(deltaGpuVsDirect[numParticles - 1], 1e-2);
}
