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
 *
 * Uses GPU octree build (TreeBuilder) to keep all data on device.
 */

#include <cstdio>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/treebuilder.cuh"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace cstone;

TEST(CartesianFMM, ProfileDualTraversal)
{
    using T       = double;
    using KeyType = uint64_t;

    unsigned            N          = 100000000;
    float               theta      = 0.5f;
    float               G          = 1.0f;
    unsigned            bucketSize = 64;
    cstone::Box<double> box(-1, 1);

    // Generate particles on CPU (SFC-sorted)
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(N, box);
    coordinates.adjustH(2, 5);

    // Upload to GPU
    thrust::device_vector<T> d_x(coordinates.x().begin(), coordinates.x().end());
    thrust::device_vector<T> d_y(coordinates.y().begin(), coordinates.y().end());
    thrust::device_vector<T> d_z(coordinates.z().begin(), coordinates.z().end());
    thrust::device_vector<T> d_m(N, T(1) / N);
    thrust::device_vector<T> d_h(coordinates.h().begin(), coordinates.h().end());

    // GPU octree build
    ryoanji::TreeBuilder<KeyType> treeBuilder(bucketSize);
    int numSources = treeBuilder.update(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), N, box);
    // x,y,z now SFC-sorted on device; h,m stay aligned (input was already SFC-sorted)

    unsigned              highestLevel = treeBuilder.maxTreeLevel();
    const TreeNodeIndex*  levelRange   = treeBuilder.levelRange();
    TreeNodeIndex         numLeaves    = treeBuilder.numLeafNodes();
    std::span<const TreeNodeIndex> levelRangeSpan(levelRange, highestLevel + 2);

    // GPU source centers + MAC
    thrust::device_vector<SourceCenterType<T>> d_centers(numSources);
    cstone::computeLeafSourceCenterGpu(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                       treeBuilder.leafToInternal(), numLeaves,
                                       treeBuilder.layout(), rawPtr(d_centers));
    cstone::upsweepCentersGpu(highestLevel, levelRange,
                              treeBuilder.childOffsets(), rawPtr(d_centers));
    cstone::setMacGpu(treeBuilder.nodeKeys(), TreeNodeIndex(numSources), rawPtr(d_centers), 1.0f / theta, box);

    float invTheta = 1.0f / theta;

    // Acceleration output buffers
    thrust::device_vector<T> d_ax(N), d_ay(N), d_az(N);

    auto runOnce = [&](fmm::FmmGpuStats* stats) -> T
    {
        checkGpuErrors(cudaMemset(rawPtr(d_ax), 0, N * sizeof(T)));
        checkGpuErrors(cudaMemset(rawPtr(d_ay), 0, N * sizeof(T)));
        checkGpuErrors(cudaMemset(rawPtr(d_az), 0, N * sizeof(T)));
        T egrav = 0;

        fmm::computeGravityFMMGpu<fmm::DirectionalMac>(
            treeBuilder.nodeKeys(), treeBuilder.childOffsets(),
            treeBuilder.internalToLeaf(), treeBuilder.leafToInternal(),
            treeBuilder.layout(), rawPtr(d_centers),
            levelRangeSpan, TreeNodeIndex(numSources), numLeaves,
            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m),
            box, G, invTheta, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az),
            &egrav, N, stats);

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
