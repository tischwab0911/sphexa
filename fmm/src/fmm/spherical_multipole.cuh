/*
 * Spherical harmonic multipole GPU kernels for FMM
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU orchestration for spherical harmonic FMM
 *
 * All computational kernels (P2M, M2M, M2L, L2L, L2P) are HOST_DEVICE_FUN
 * and defined in spherical_multipole.hpp. This file provides GPU launch
 * wrappers, table upload, and the dual-traversal kernel.
 */

#pragma once

#include <cstdio>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal_gpu.cuh"

#include "ryoanji/nbody/kernel.hpp"
#include "ryoanji/nbody/types.h"

#include "spherical_multipole.hpp"

namespace fmm
{

// ---------------------------------------------------------------------------
// GPU tables
// ---------------------------------------------------------------------------

struct GpuSphericalTables
{
    double*          prefactor; // [4*P*P] on device
    double*          Anm;       // [4*P*P] on device
    Complex<double>* Cnm;       // [P*P*P*P] on device
};

inline GpuSphericalTables uploadSphericalTables()
{
    constexpr int PP  = ExpansionOrder;
    constexpr int PP2 = PP * PP;
    constexpr int PP4 = PP2 * PP2;

    const auto& tab = SphericalTables<PP>::instance();

    GpuSphericalTables gpu;
    checkGpuErrors(cudaMalloc(&gpu.prefactor, 4 * PP2 * sizeof(double)));
    checkGpuErrors(cudaMalloc(&gpu.Anm, 4 * PP2 * sizeof(double)));
    checkGpuErrors(cudaMalloc(&gpu.Cnm, PP4 * sizeof(Complex<double>)));

    checkGpuErrors(cudaMemcpy(gpu.prefactor, tab.prefactor.data(), 4 * PP2 * sizeof(double), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(gpu.Anm, tab.Anm.data(), 4 * PP2 * sizeof(double), cudaMemcpyHostToDevice));
    // std::complex<double> and thrust::complex<double> have identical layout
    checkGpuErrors(cudaMemcpy(gpu.Cnm, tab.Cnm.data(), PP4 * sizeof(Complex<double>), cudaMemcpyHostToDevice));

    return gpu;
}

inline void freeSphericalTables(GpuSphericalTables& gpu)
{
    cudaFree(gpu.prefactor);
    cudaFree(gpu.Anm);
    cudaFree(gpu.Cnm);
    gpu.prefactor = nullptr;
    gpu.Anm       = nullptr;
    gpu.Cnm       = nullptr;
}

// ---------------------------------------------------------------------------
// P2M kernel — compute leaf multipoles on GPU
// ---------------------------------------------------------------------------

template<int TPL, class T>
__global__ void computeLeafMultipolesGpuKernel(const T* x, const T* y, const T* z, const T* m,
                                                const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                                const LocalIndex* layout, const Vec4<T>* centers,
                                                SphericalMultipole<T>* multipoles, GpuSphericalTables tables)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;
    TreeNodeIndex internalIdx;

    SphericalMultipole<T> mp_loc;
    for (auto& v : mp_loc)
        v = Complex<T>(0, 0);

    if (leafIdx < numLeaves)
    {
        internalIdx = leafToInternal[leafIdx];
        auto com    = centers[internalIdx];
        P2M<TPL>(x, y, z, m, layout[leafIdx] + threadIdx.x % TPL, layout[leafIdx + 1], com, mp_loc,
                 tables.prefactor);
    }

    // Warp-level reduction across TPL threads
    constexpr int mpNumElements = Nterm<ExpansionOrder>;
#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
#pragma unroll
        for (int mi = 0; mi < mpNumElements; ++mi)
            mp_loc[mi] = mp_loc[mi] + cstone::shflDownSync(mp_loc[mi], offset);
    }

    if (tid % TPL == 0 && leafIdx < numLeaves) { multipoles[internalIdx] = mp_loc; }
}

template<class T>
void computeLeafMultipolesGpu(const T* d_x, const T* d_y, const T* d_z, const T* d_m,
                               const TreeNodeIndex* d_leafToInternal, TreeNodeIndex numLeaves,
                               const LocalIndex* d_layout, const Vec4<T>* d_centers,
                               SphericalMultipole<T>* d_multipoles, const GpuSphericalTables& tables)
{
    constexpr int numThreads    = 256;
    constexpr int threadsPerLeaf = 8;
    int           numBlocks     = cstone::iceil(threadsPerLeaf * numLeaves, numThreads);

    if (numBlocks)
    {
        computeLeafMultipolesGpuKernel<threadsPerLeaf>
            <<<numBlocks, numThreads>>>(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers,
                                        d_multipoles, tables);
    }
}

// ---------------------------------------------------------------------------
// M2M upsweep kernel
// ---------------------------------------------------------------------------

template<class T>
__global__ void upsweepMultipolesGpuKernel(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                            const TreeNodeIndex* childOffsets, const Vec4<T>* centers,
                                            SphericalMultipole<T>* multipoles, GpuSphericalTables tables)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int     cellIdx = tid / 8 + firstCell;

    TreeNodeIndex firstChild = 0;
    if (cellIdx < lastCell) { firstChild = childOffsets[cellIdx]; }

    SphericalMultipole<T> Mout;
    for (auto& v : Mout)
        v = Complex<T>(0, 0);

    if (firstChild)
    {
        int child = firstChild + threadIdx.x % 8;
        M2M(centers[cellIdx], centers[child], multipoles[child], Mout, tables.prefactor, tables.Anm);
    }

    constexpr int mpNumElements = Nterm<ExpansionOrder>;
#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2)
    {
#pragma unroll
        for (int mi = 0; mi < mpNumElements; ++mi)
            Mout[mi] = Mout[mi] + cstone::shflDownSync(Mout[mi], offset);
    }

    if (firstChild && threadIdx.x % 8 == 0) { multipoles[cellIdx] = Mout; }
}

template<class T>
void upsweepMultipolesGpu(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* d_childOffsets,
                           const Vec4<T>* d_centers, SphericalMultipole<T>* d_multipoles,
                           const GpuSphericalTables& tables)
{
    constexpr int numThreads = 256;
    int           numLevels  = int(levelRange.size()) - 2;

    for (int level = numLevels; level >= 0; --level)
    {
        TreeNodeIndex firstCell = levelRange[level];
        TreeNodeIndex lastCell  = levelRange[level + 1];
        if (lastCell > firstCell)
        {
            upsweepMultipolesGpuKernel<<<cstone::iceil(8 * (lastCell - firstCell), numThreads), numThreads>>>(
                firstCell, lastCell, d_childOffsets, d_centers, d_multipoles, tables);
        }
    }
}

// ---------------------------------------------------------------------------
// L2L downsweep kernel
// ---------------------------------------------------------------------------

template<class T>
__global__ void l2lKernel(TreeNodeIndex start, TreeNodeIndex end, const TreeNodeIndex* childOffsets,
                          const Vec4<T>* centers, SphericalLocalExpansion<T>* locals, GpuSphericalTables tables)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end) return;

    TreeNodeIndex firstChild = childOffsets[i];
    if (firstChild == 0) return; // leaf node

    for (int c = firstChild; c < firstChild + 8; ++c)
    {
        L2L(centers[i], centers[c], locals[i], locals[c], tables.prefactor, tables.Anm);
    }
}

template<class T>
void downsweepLocalExpansionsGpu(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* d_childOffsets,
                                  const Vec4<T>* d_centers, SphericalLocalExpansion<T>* d_locals,
                                  const GpuSphericalTables& tables)
{
    constexpr int numThreads = 256;
    int           numLevels  = int(levelRange.size()) - 1;

    for (int level = 0; level < numLevels; ++level)
    {
        TreeNodeIndex start = levelRange[level];
        TreeNodeIndex end   = levelRange[level + 1];
        TreeNodeIndex count = end - start;
        if (count > 0)
        {
            l2lKernel<<<(count + numThreads - 1) / numThreads, numThreads>>>(start, end, d_childOffsets, d_centers,
                                                                              d_locals, tables);
        }
    }
}

// ---------------------------------------------------------------------------
// L2P kernel
// ---------------------------------------------------------------------------

template<class T>
__global__ void l2pKernel(TreeNodeIndex firstLeaf, TreeNodeIndex lastLeaf, const TreeNodeIndex* leafToInternalMap,
                          const LocalIndex* layout, const Vec4<T>* centers,
                          const SphericalLocalExpansion<T>* locals, const T* x, const T* y, const T* z,
                          T* ppot, T* pax, T* pay, T* paz, LocalIndex firstTarget, GpuSphericalTables tables)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstLeaf;
    if (leafIdx >= lastLeaf) return;

    TreeNodeIndex nodeIdx = leafToInternalMap[leafIdx];
    LocalIndex    first   = layout[leafIdx];
    LocalIndex    last    = layout[leafIdx + 1];
    const auto&   L       = locals[nodeIdx];
    Vec3<T>       center  = util::makeVec3(centers[nodeIdx]);

    for (LocalIndex t = first; t < last; ++t)
    {
        LocalIndex ti = t - firstTarget;
        Vec4<T>    acc{ppot[ti], pax[ti], pay[ti], paz[ti]};
        Vec3<T>    target{x[t], y[t], z[t]};
        acc      = L2P(acc, target, center, L, tables.prefactor);
        ppot[ti] = acc[0];
        pax[ti]  = acc[1];
        pay[ti]  = acc[2];
        paz[ti]  = acc[3];
    }
}

// ---------------------------------------------------------------------------
// Dual traversal kernel
// ---------------------------------------------------------------------------

using FmmTravConfig = cstone::TraversalConfig<1024, 224, 704, 512, 160, 96, 192, 640, 224, 32, 1>;

struct FmmDualConfig
{
    static constexpr unsigned numWarps          = 7;
    static constexpr unsigned numThreadsPerBlock = numWarps * cstone::GpuConfig::warpSize;
    static constexpr unsigned kBlocksPerCluster = 8;
};

template<MacVariant macType, int numWarps, class T>
__global__ void fmmDualTraversalKernel(const TreeNodeIndex* __restrict__ childOffsets,
                                       const Vec3<T>* __restrict__ geoCenters,
                                       const Vec3<T>* __restrict__ geoSizes,
                                       const Vec4<T>* __restrict__ centers,
                                       const SphericalMultipole<T>* __restrict__ multipoles,
                                       SphericalLocalExpansion<T>* __restrict__ locals,
                                       const TreeNodeIndex* __restrict__ internalToLeaf,
                                       const LocalIndex* __restrict__ layout,
                                       const T* __restrict__ x, const T* __restrict__ y,
                                       const T* __restrict__ z, const T* __restrict__ h, const T* __restrict__ m,
                                       T* __restrict__ ppot, T* __restrict__ pax, T* __restrict__ pay,
                                       T* __restrict__ paz, LocalIndex firstTarget,
                                       GpuSphericalTables tables,
                                       cstone::GlobalWorkQueue gq, cstone::GlobalTraversalQueue tq,
                                       unsigned* nProd)
{
    auto continuation = [centers, geoCenters, geoSizes]
        __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        if constexpr (macType == DirectionalMac)
        {
            return cstone::evaluateMacM2L(
                util::makeVec3(centers[a]), centers[a][3],
                util::makeVec3(centers[b]), centers[b][3]);
        }
        else
        {
            return cstone::evaluateMac(util::makeVec3(centers[a]), centers[a][3],
                                        geoCenters[b], geoSizes[b]);
        }
    };

    auto m2l = [centers, multipoles, locals, tables] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        // Compute M2L into thread-local buffer, then atomic-add into shared local expansion
        SphericalLocalExpansion<T> localBuf;
        for (auto& v : localBuf)
            v = Complex<T>(0, 0);
        M2L(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], localBuf, tables.prefactor,
            tables.Anm, tables.Cnm);
        for (int i = 0; i < Nterm<ExpansionOrder>; ++i)
        {
            atomicAdd(reinterpret_cast<T*>(&locals[a][i]), localBuf[i].real());
            atomicAdd(reinterpret_cast<T*>(&locals[a][i]) + 1, localBuf[i].imag());
        }
    };

    auto p2p = [internalToLeaf, layout, x, y, z, h, m, ppot, pax, pay, paz,
                firstTarget] __device__(unsigned p2pMask, TreeNodeIndex a, TreeNodeIndex b)
    {
        unsigned lane = threadIdx.x % cstone::GpuConfig::warpSize;
        if (!((p2pMask >> lane) & 1u)) return;

        TreeNodeIndex aLeaf = internalToLeaf[a];
        TreeNodeIndex bLeaf = internalToLeaf[b];

        LocalIndex aFirst = layout[aLeaf];
        LocalIndex aLast  = layout[aLeaf + 1];
        LocalIndex bFirst = layout[bLeaf];
        LocalIndex bLast  = layout[bLeaf + 1];

        for (LocalIndex t = aFirst; t < aLast; ++t)
        {
            Vec4<T> acc{0, 0, 0, 0};
            Vec3<T> target{x[t], y[t], z[t]};
            for (LocalIndex s = bFirst; s < bLast; ++s)
            {
                acc = ryoanji::P2P(acc, target, Vec3<T>{x[s], y[s], z[s]}, m[s], h[t], h[s]);
            }
            LocalIndex ti = t - firstTarget;
            atomicAdd(&ppot[ti], acc[0]);
            atomicAdd(&pax[ti], acc[1]);
            atomicAdd(&pay[ti], acc[2]);
            atomicAdd(&paz[ti], acc[3]);
        }
    };

    cstone::dualTraversalGPU<numWarps, FmmTravConfig>(childOffsets, TreeNodeIndex(0), TreeNodeIndex(0), gq, tq, nProd,
                                                      continuation, m2l, p2p);
}

// ---------------------------------------------------------------------------
// Host orchestration — host-pointer overload
// ---------------------------------------------------------------------------

template<MacVariant macType = ScalarMac, class T, class KeyType>
void computeGravityFMMGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                           const TreeNodeIndex* internalToLeaf,
                           std::span<const TreeNodeIndex> leafToInternalMap,
                           std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                           const SphericalMultipole<T>* /*multipoles*/, const LocalIndex* layout,
                           TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                           const T* z, const T* h, const T* m, const cstone::Box<T>& box, float G,
                           float invTheta, T* ugrav, T* ax, T* ay, T* az, T* ugravTot, LocalIndex numParticles)
{
    TreeNodeIndex numNodes     = levelRange.back();
    TreeNodeIndex numLeaves    = TreeNodeIndex(leafToInternalMap.size());
    LocalIndex    firstTarget  = layout[firstLeafIndex];
    LocalIndex    lastTarget   = layout[lastLeafIndex];
    LocalIndex    numTargets   = lastTarget - firstTarget;

    // 1. Upload SphericalTables
    GpuSphericalTables tables = uploadSphericalTables();

    // 2. Compute geometric centers/sizes from SFC prefixes
    std::vector<Vec3<T>> h_geoCenters(numNodes);
    std::vector<Vec3<T>> h_geoSizes(numNodes);
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        KeyType  prefix   = prefixes[i];
        KeyType  startKey = cstone::decodePlaceholderBit(prefix);
        unsigned level    = cstone::decodePrefixLength(prefix) / 3;
        auto     nodeBox  = cstone::sfcIBox(cstone::sfcKey(startKey), level);
        auto [center, sz] = cstone::centerAndSize<KeyType>(nodeBox, box);
        h_geoCenters[i]   = center;
        h_geoSizes[i]     = sz * T(invTheta);
    }

    // 3. Upload tree structure arrays
    Vec3<T>*       d_geoCenters;
    Vec3<T>*       d_geoSizes;
    TreeNodeIndex* d_childOffsets;
    TreeNodeIndex* d_internalToLeaf;
    TreeNodeIndex* d_leafToInternal;
    LocalIndex*    d_layout;
    Vec4<T>*       d_centers;

    checkGpuErrors(cudaMalloc(&d_geoCenters, numNodes * sizeof(Vec3<T>)));
    checkGpuErrors(cudaMalloc(&d_geoSizes, numNodes * sizeof(Vec3<T>)));
    checkGpuErrors(cudaMalloc(&d_childOffsets, (numNodes + 1) * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_internalToLeaf, numNodes * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_leafToInternal, numLeaves * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_layout, (numLeaves + 1) * sizeof(LocalIndex)));
    checkGpuErrors(cudaMalloc(&d_centers, numNodes * sizeof(Vec4<T>)));

    checkGpuErrors(
        cudaMemcpy(d_geoCenters, h_geoCenters.data(), numNodes * sizeof(Vec3<T>), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_geoSizes, h_geoSizes.data(), numNodes * sizeof(Vec3<T>), cudaMemcpyHostToDevice));
    checkGpuErrors(
        cudaMemcpy(d_childOffsets, childOffsets, (numNodes + 1) * sizeof(TreeNodeIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(
        cudaMemcpy(d_internalToLeaf, internalToLeaf, numNodes * sizeof(TreeNodeIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_leafToInternal, leafToInternalMap.data(), numLeaves * sizeof(TreeNodeIndex),
                              cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_layout, layout, (numLeaves + 1) * sizeof(LocalIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_centers, centers, numNodes * sizeof(Vec4<T>), cudaMemcpyHostToDevice));

    // 4. Upload particle arrays
    T *d_x, *d_y, *d_z, *d_h, *d_m;
    checkGpuErrors(cudaMalloc(&d_x, numParticles * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_y, numParticles * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_z, numParticles * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_h, numParticles * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_m, numParticles * sizeof(T)));

    checkGpuErrors(cudaMemcpy(d_x, x, numParticles * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_y, y, numParticles * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_z, z, numParticles * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_h, h, numParticles * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_m, m, numParticles * sizeof(T), cudaMemcpyHostToDevice));

    // 5. Allocate device multipoles and compute P2M + M2M on GPU
    SphericalMultipole<T>* d_multipoles;
    checkGpuErrors(cudaMalloc(&d_multipoles, numNodes * sizeof(SphericalMultipole<T>)));
    checkGpuErrors(cudaMemset(d_multipoles, 0, numNodes * sizeof(SphericalMultipole<T>)));

    computeLeafMultipolesGpu(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers, d_multipoles,
                             tables);

    upsweepMultipolesGpu(levelRange, d_childOffsets, d_centers, d_multipoles, tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // 6. Allocate + zero-init device locals and particle accumulators
    SphericalLocalExpansion<T>* d_locals;
    checkGpuErrors(cudaMalloc(&d_locals, numNodes * sizeof(SphericalLocalExpansion<T>)));
    checkGpuErrors(cudaMemset(d_locals, 0, numNodes * sizeof(SphericalLocalExpansion<T>)));

    T *d_ppot, *d_pax, *d_pay, *d_paz;
    checkGpuErrors(cudaMalloc(&d_ppot, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_pax, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_pay, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_paz, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_ppot, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_pax, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_pay, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_paz, 0, numTargets * sizeof(T)));

    // 7. Allocate global work queues for dual traversal
    constexpr unsigned gChunk = FmmTravConfig::chunkSize;
    constexpr unsigned gSegs  = 2048;
    constexpr unsigned gCap   = gSegs * gChunk;

    TreeNodeIndex* d_gA;
    TreeNodeIndex* d_gB;
    int*           d_gIsP2P;
    unsigned*      d_wHead;
    unsigned*      d_rHead;
    unsigned*      d_segCount;
    unsigned*      d_segR;
    unsigned*      d_nProd;

    checkGpuErrors(cudaMalloc(&d_gA, gCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_gB, gCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_gIsP2P, gCap * sizeof(int)));
    checkGpuErrors(cudaMalloc(&d_wHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_rHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_segCount, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_segR, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_nProd, sizeof(unsigned)));

    cstone::GlobalWorkQueue gq{d_gA, d_gB, d_gIsP2P, d_wHead, d_rHead, d_segCount, d_segR, gSegs};

    constexpr unsigned tChunk = FmmTravConfig::travChunkSize;
    constexpr unsigned tSegs  = 2048;
    constexpr unsigned tCap   = tSegs * tChunk;

    TreeNodeIndex* d_tA;
    TreeNodeIndex* d_tB;
    unsigned*      d_twHead;
    unsigned*      d_trHead;
    unsigned*      d_tsegR;

    checkGpuErrors(cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_twHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_trHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_tsegR, tSegs * sizeof(unsigned)));

    cstone::GlobalTraversalQueue tq{d_tA, d_tB, d_twHead, d_trHead, d_tsegR, tSegs};

    // 8. Reset queues
    checkGpuErrors(cudaMemset(d_wHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_rHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segR, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_twHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_trHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned)));

    // 9. Configure cluster launch
    constexpr unsigned kBlocksPerCluster = FmmDualConfig::kBlocksPerCluster;
    constexpr unsigned numWarps          = FmmDualConfig::numWarps;
    constexpr unsigned threadsPerBlock   = FmmDualConfig::numThreadsPerBlock;

    unsigned maxBlocks = cstone::maxConcurrentBlocks(
        fmmDualTraversalKernel<macType, numWarps, T>, threadsPerBlock, kBlocksPerCluster);
    unsigned totalBlocks = maxBlocks;

    printf("[FMM] Launching %u blocks (%u clusters of %u)\n",
           totalBlocks, totalBlocks / kBlocksPerCluster, kBlocksPerCluster);

    cudaLaunchConfig_t    dualCfg{};
    dualCfg.gridDim  = {totalBlocks, 1, 1};
    dualCfg.blockDim = {threadsPerBlock, 1, 1};
    cudaLaunchAttribute dualAttr{};
    dualAttr.id               = cudaLaunchAttributeClusterDimension;
    dualAttr.val.clusterDim.x = kBlocksPerCluster;
    dualAttr.val.clusterDim.y = 1;
    dualAttr.val.clusterDim.z = 1;
    dualCfg.attrs    = &dualAttr;
    dualCfg.numAttrs = 1;

    checkGpuErrors(cudaMemset(d_nProd, 0, sizeof(unsigned)));

    // 10. Launch dual traversal
    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, fmmDualTraversalKernel<macType, numWarps, T>, d_childOffsets,
                                      d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals, d_internalToLeaf,
                                      d_layout, d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget,
                                      tables, gq, tq, d_nProd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 11. L2L downsweep
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals, tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // 12. L2P
    {
        constexpr int numThreads = 256;
        TreeNodeIndex numLeafRange = lastLeafIndex - firstLeafIndex;
        if (numLeafRange > 0)
        {
            l2pKernel<<<(numLeafRange + numThreads - 1) / numThreads, numThreads>>>(
                firstLeafIndex, lastLeafIndex, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z, d_ppot,
                d_pax, d_pay, d_paz, firstTarget, tables);
            checkGpuErrors(cudaDeviceSynchronize());
        }
    }

    // 13. Download results
    std::vector<T> h_ppot(numTargets), h_pax(numTargets), h_pay(numTargets), h_paz(numTargets);
    checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pax.data(), d_pax, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pay.data(), d_pay, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_paz.data(), d_paz, numTargets * sizeof(T), cudaMemcpyDeviceToHost));

    // 14. Apply G scaling and accumulate into output arrays
    T ugravLoc = 0;
    for (LocalIndex t = firstTarget; t < lastTarget; ++t)
    {
        LocalIndex ti = t - firstTarget;
        auto       u  = G * m[t] * h_ppot[ti];
        ugravLoc += u;
        if (ugrav) { ugrav[t] += u; }
        ax[t] += G * h_pax[ti];
        ay[t] += G * h_pay[ti];
        az[t] += G * h_paz[ti];
    }
    *ugravTot += T(0.5) * ugravLoc;

    // 15. Free device memory
    freeSphericalTables(tables);

    cudaFree(d_geoCenters);
    cudaFree(d_geoSizes);
    cudaFree(d_childOffsets);
    cudaFree(d_internalToLeaf);
    cudaFree(d_leafToInternal);
    cudaFree(d_layout);
    cudaFree(d_centers);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_m);

    cudaFree(d_multipoles);
    cudaFree(d_locals);

    cudaFree(d_ppot);
    cudaFree(d_pax);
    cudaFree(d_pay);
    cudaFree(d_paz);

    cudaFree(d_gA);
    cudaFree(d_gB);
    cudaFree(d_gIsP2P);
    cudaFree(d_wHead);
    cudaFree(d_rHead);
    cudaFree(d_segCount);
    cudaFree(d_segR);
    cudaFree(d_nProd);

    cudaFree(d_tA);
    cudaFree(d_tB);
    cudaFree(d_twHead);
    cudaFree(d_trHead);
    cudaFree(d_tsegR);
}

// ---------------------------------------------------------------------------
// Helper kernel for GPU-native spherical path
// ---------------------------------------------------------------------------

template<class T>
__global__ void sphApplyGScalingKernel(LocalIndex n, float G,
                                        const T* pax, const T* pay, const T* paz,
                                        T* ax, T* ay, T* az)
{
    LocalIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        ax[i] += T(G) * pax[i];
        ay[i] += T(G) * pay[i];
        az[i] += T(G) * paz[i];
    }
}

// ---------------------------------------------------------------------------
// Helper kernel: scale Vec3 array by a scalar factor
// ---------------------------------------------------------------------------

template<class T>
__global__ void sphScaleVec3Kernel(Vec3<T>* data, TreeNodeIndex n, T factor)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i][0] *= factor;
        data[i][1] *= factor;
        data[i][2] *= factor;
    }
}

// ---------------------------------------------------------------------------
// GPU-native overload — takes device pointers, no CPU upload/download
// ---------------------------------------------------------------------------

template<MacVariant macType = ScalarMac, class T, class KeyType>
void computeGravityFMMGpu(
    const KeyType* d_prefixes, const TreeNodeIndex* d_childOffsets,
    const TreeNodeIndex* d_internalToLeaf, const TreeNodeIndex* d_leafToInternal,
    const LocalIndex* d_layout, const cstone::SourceCenterType<T>* d_centers,
    std::span<const TreeNodeIndex> levelRange,
    TreeNodeIndex numNodes, TreeNodeIndex numLeaves,
    const T* d_x, const T* d_y, const T* d_z, const T* d_h, const T* d_m,
    const cstone::Box<T>& box, float G, float invTheta,
    T* d_ax, T* d_ay, T* d_az,
    T* ugravTot, LocalIndex numParticles)
{
    LocalIndex firstTarget = 0;
    LocalIndex numTargets  = numParticles;

    // 1. Upload SphericalTables
    GpuSphericalTables tables = uploadSphericalTables();

    // 2. Compute geometric centers/sizes on GPU from SFC prefixes
    Vec3<T>* d_geoCenters;
    Vec3<T>* d_geoSizes;
    checkGpuErrors(cudaMalloc(&d_geoCenters, numNodes * sizeof(Vec3<T>)));
    checkGpuErrors(cudaMalloc(&d_geoSizes, numNodes * sizeof(Vec3<T>)));
    cstone::computeGeoCentersGpu(d_prefixes, numNodes, d_geoCenters, d_geoSizes, box);

    // Scale geo sizes by invTheta for scalar MAC
    if constexpr (macType == ScalarMac)
    {
        int nt = 256;
        int nb = cstone::iceil(numNodes, nt);
        if (nb) { sphScaleVec3Kernel<<<nb, nt>>>(d_geoSizes, numNodes, T(invTheta)); }
    }

    // 3. Allocate device multipoles and compute P2M + M2M on GPU
    SphericalMultipole<T>* d_multipoles;
    checkGpuErrors(cudaMalloc(&d_multipoles, numNodes * sizeof(SphericalMultipole<T>)));
    checkGpuErrors(cudaMemset(d_multipoles, 0, numNodes * sizeof(SphericalMultipole<T>)));

    computeLeafMultipolesGpu(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers, d_multipoles,
                             tables);
    upsweepMultipolesGpu(levelRange, d_childOffsets, d_centers, d_multipoles, tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // 4. Allocate + zero-init device locals and particle accumulators
    SphericalLocalExpansion<T>* d_locals;
    checkGpuErrors(cudaMalloc(&d_locals, numNodes * sizeof(SphericalLocalExpansion<T>)));
    checkGpuErrors(cudaMemset(d_locals, 0, numNodes * sizeof(SphericalLocalExpansion<T>)));

    T *d_ppot, *d_pax, *d_pay, *d_paz;
    checkGpuErrors(cudaMalloc(&d_ppot, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_pax, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_pay, numTargets * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_paz, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_ppot, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_pax, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_pay, 0, numTargets * sizeof(T)));
    checkGpuErrors(cudaMemset(d_paz, 0, numTargets * sizeof(T)));

    // 5. Allocate global work queues for dual traversal
    constexpr unsigned gChunk = FmmTravConfig::chunkSize;
    constexpr unsigned gSegs  = 2048;
    constexpr unsigned gCap   = gSegs * gChunk;

    TreeNodeIndex* d_gA;
    TreeNodeIndex* d_gB;
    int*           d_gIsP2P;
    unsigned*      d_wHead;
    unsigned*      d_rHead;
    unsigned*      d_segCount;
    unsigned*      d_segR;
    unsigned*      d_nProd;

    checkGpuErrors(cudaMalloc(&d_gA, gCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_gB, gCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_gIsP2P, gCap * sizeof(int)));
    checkGpuErrors(cudaMalloc(&d_wHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_rHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_segCount, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_segR, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_nProd, sizeof(unsigned)));

    cstone::GlobalWorkQueue gq{d_gA, d_gB, d_gIsP2P, d_wHead, d_rHead, d_segCount, d_segR, gSegs};

    constexpr unsigned tChunk = FmmTravConfig::travChunkSize;
    constexpr unsigned tSegs  = 2048;
    constexpr unsigned tCap   = tSegs * tChunk;

    TreeNodeIndex* d_tA;
    TreeNodeIndex* d_tB;
    unsigned*      d_twHead;
    unsigned*      d_trHead;
    unsigned*      d_tsegR;

    checkGpuErrors(cudaMalloc(&d_tA, tCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_tB, tCap * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_twHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_trHead, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_tsegR, tSegs * sizeof(unsigned)));

    cstone::GlobalTraversalQueue tq{d_tA, d_tB, d_twHead, d_trHead, d_tsegR, tSegs};

    // 6. Reset queues
    checkGpuErrors(cudaMemset(d_wHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_rHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segR, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_twHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_trHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned)));

    // 7. Configure cluster launch
    constexpr unsigned kBlocksPerCluster = FmmDualConfig::kBlocksPerCluster;
    constexpr unsigned numWarps          = FmmDualConfig::numWarps;
    constexpr unsigned threadsPerBlock   = FmmDualConfig::numThreadsPerBlock;

    unsigned maxBlocks = cstone::maxConcurrentBlocks(
        fmmDualTraversalKernel<macType, numWarps, T>, threadsPerBlock, kBlocksPerCluster);
    unsigned totalBlocks = maxBlocks;

    cudaLaunchConfig_t    dualCfg{};
    dualCfg.gridDim  = {totalBlocks, 1, 1};
    dualCfg.blockDim = {threadsPerBlock, 1, 1};
    cudaLaunchAttribute dualAttr{};
    dualAttr.id               = cudaLaunchAttributeClusterDimension;
    dualAttr.val.clusterDim.x = kBlocksPerCluster;
    dualAttr.val.clusterDim.y = 1;
    dualAttr.val.clusterDim.z = 1;
    dualCfg.attrs    = &dualAttr;
    dualCfg.numAttrs = 1;

    checkGpuErrors(cudaMemset(d_nProd, 0, sizeof(unsigned)));

    // 8. Launch dual traversal
    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, fmmDualTraversalKernel<macType, numWarps, T>, d_childOffsets,
                                      d_geoCenters, d_geoSizes, d_centers, d_multipoles, d_locals, d_internalToLeaf,
                                      d_layout, d_x, d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget,
                                      tables, gq, tq, d_nProd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 9. L2L downsweep
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals, tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // 10. L2P
    {
        constexpr int numThreads = 256;
        if (numLeaves > 0)
        {
            l2pKernel<<<(numLeaves + numThreads - 1) / numThreads, numThreads>>>(
                0, numLeaves, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z, d_ppot,
                d_pax, d_pay, d_paz, firstTarget, tables);
            checkGpuErrors(cudaDeviceSynchronize());
        }
    }

    // 11. Apply G-scaling on GPU and accumulate into caller's output buffers
    {
        int nt = 256;
        int nb = cstone::iceil(numTargets, nt);
        if (nb)
        {
            sphApplyGScalingKernel<<<nb, nt>>>(numTargets, G, d_pax, d_pay, d_paz,
                                               d_ax, d_ay, d_az);
        }
    }

    // 12. Download potential sum for ugravTot (small scalar)
    if (ugravTot)
    {
        std::vector<T> h_ppot(numTargets);
        std::vector<T> h_m(numTargets);
        checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
        checkGpuErrors(cudaMemcpy(h_m.data(), d_m, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
        T ugravLoc = 0;
        for (LocalIndex i = 0; i < numTargets; ++i)
            ugravLoc += G * h_m[i] * h_ppot[i];
        *ugravTot += T(0.5) * ugravLoc;
    }

    checkGpuErrors(cudaDeviceSynchronize());

    // 13. Free internally-allocated temporaries (caller owns input device pointers)
    freeSphericalTables(tables);

    cudaFree(d_geoCenters);
    cudaFree(d_geoSizes);

    cudaFree(d_multipoles);
    cudaFree(d_locals);

    cudaFree(d_ppot);
    cudaFree(d_pax);
    cudaFree(d_pay);
    cudaFree(d_paz);

    cudaFree(d_gA);
    cudaFree(d_gB);
    cudaFree(d_gIsP2P);
    cudaFree(d_wHead);
    cudaFree(d_rHead);
    cudaFree(d_segCount);
    cudaFree(d_segR);
    cudaFree(d_nProd);

    cudaFree(d_tA);
    cudaFree(d_tB);
    cudaFree(d_twHead);
    cudaFree(d_trHead);
    cudaFree(d_tsegR);
}

} // namespace fmm
