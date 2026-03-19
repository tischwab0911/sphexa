/*
 * Cartesian quadrupole FMM GPU vs GPU Barnes-Hut test
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test GPU Cartesian quadrupole FMM against GPU Barnes-Hut reference
 *
 * Builds an octree, runs the GPU FMM (dual traversal) and the GPU
 * Barnes-Hut tree walk (both at quadrupole order), and compares results.
 */

#include <chrono>
#include <numeric>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups_gpu.h"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

TEST(CartesianQpoleFMM, GpuFmmVsGpuBH)
{
    using T             = double;
    using KeyType       = uint64_t;
    using FmmMpole      = fmm::CartesianMultipole<T>;
    using BhMpole       = ryoanji::CartesianQuadrupole<T>;

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

    // ----- GPU FMM path -----
    // Compute FMM multipoles (P2M + M2M upsweep via ADL on fmm::CartesianMultipole)
    std::vector<FmmMpole> fmmMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   fmmMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), fmmMultipoles.data());

    std::vector<T> gpuAx(numParticles, 0);
    std::vector<T> gpuAy(numParticles, 0);
    std::vector<T> gpuAz(numParticles, 0);
    T              gpuEgrav = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    fmm::computeGravityFMMGpu<T, KeyType>(
        octree.prefixes.data(), octree.childOffsets.data(), octree.internalToLeaf.data(), toInternal,
        std::span<const TreeNodeIndex>(octree.levelRange), centers.data(), fmmMultipoles.data(), layout.data(), 0,
        octree.numLeafNodes, x, y, z, h, masses.data(), box, theta, G, (T*)nullptr, gpuAx.data(), gpuAy.data(),
        gpuAz.data(), &gpuEgrav, numParticles);
    auto   t1          = std::chrono::high_resolution_clock::now();
    double gpuElapsed  = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "GPU FMM (Cartesian quadrupole) for " << numParticles << " particles: " << gpuElapsed << " s"
              << std::endl;

    // ----- GPU Barnes-Hut path -----
    // Compute BH multipoles (P2M + M2M upsweep via ADL on ryoanji::CartesianQuadrupole)
    std::vector<BhMpole> bhMultipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(),
                                   bhMultipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), bhMultipoles.data());

    // Upload to device
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

    t0 = std::chrono::high_resolution_clock::now();
    T bhEgrav = ryoanji::traverse(
        groups.view(), octree.childOffsets[0],
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
        rawPtr(d_childOffsets), rawPtr(d_internalToLeaf), rawPtr(d_layout),
        rawPtr(d_centers), rawPtr(d_bhMultipoles),
        T(G), 0, ryoanji::Vec3<T>{box.lx(), box.ly(), box.lz()},
        (T*)nullptr, rawPtr(d_bhAx), rawPtr(d_bhAy), rawPtr(d_bhAz),
        thrust::raw_pointer_cast(globalPool.data()));
    t1                 = std::chrono::high_resolution_clock::now();
    double bhElapsed   = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "GPU Barnes-Hut (quadrupole) for " << numParticles << " particles: " << bhElapsed << " s"
              << std::endl;

    // Copy results back
    std::vector<T> bhAx(numParticles), bhAy(numParticles), bhAz(numParticles);
    thrust::copy(d_bhAx.begin(), d_bhAx.end(), bhAx.begin());
    thrust::copy(d_bhAy.begin(), d_bhAy.end(), bhAy.begin());
    thrust::copy(d_bhAz.begin(), d_bhAz.end(), bhAz.begin());

    // ----- Timing Comparison -----
    std::cout << "\n=== Timing Comparison ===" << std::endl;
    std::cout << "GPU FMM (dual traversal):   " << gpuElapsed << " s" << std::endl;
    std::cout << "GPU BH  (single traversal): " << bhElapsed << " s" << std::endl;
    if (gpuElapsed > 0)
        std::cout << "Speedup (BH/FMM): " << bhElapsed / gpuElapsed << "x" << std::endl;

    // ----- Accuracy Comparison -----
    std::cout << "\nGPU FMM energy: " << gpuEgrav << "  GPU BH energy: " << bhEgrav << std::endl;

    double energyRelErr = std::abs(gpuEgrav - bhEgrav) / std::abs(bhEgrav);
    std::cout << "Energy relative error: " << energyRelErr << std::endl;
    EXPECT_LT(energyRelErr, 0.05);

    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> aGpu{gpuAx[i], gpuAy[i], gpuAz[i]};
        ryoanji::Vec3<T> aBh{bhAx[i], bhAy[i], bhAz[i]};
        T                bhNorm = norm2(aBh);
        delta[i] = bhNorm > 0 ? std::sqrt(norm2(aGpu - aBh) / bhNorm) : 0;
    }

    std::sort(delta.begin(), delta.end());

    std::cout.precision(10);
    std::cout << "GPU FMM vs GPU BH:" << std::endl;
    std::cout << "  50th percentile: " << delta[numParticles / 2] << std::endl;
    std::cout << "  99th percentile: " << delta[LocalIndex(numParticles * 0.99)] << std::endl;
    std::cout << "  max error:       " << delta[numParticles - 1] << std::endl;

    EXPECT_LT(delta[LocalIndex(numParticles * 0.99)], 0.15);
    EXPECT_LT(delta[numParticles - 1], 0.5);
}
