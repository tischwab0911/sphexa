/*
 * Cartesian quadrupole FMM GPU kernels
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU port of Cartesian quadrupole FMM kernels
 *
 * Much simpler than the spherical harmonic GPU port: no complex numbers,
 * no precomputed tables, tiny expansions (8+10 reals vs 42 reals).
 * Pure-arithmetic kernels (P2M, addQuadrupole, M2L, L2L, L2P) from the
 * HOST_DEVICE_FUN-annotated .hpp are called directly from device code.
 */

#pragma once

#include <cstdio>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/traversal_gpu.cuh"

#include "ryoanji/nbody/kernel.hpp"
#include "ryoanji/nbody/types.h"

#include "cartesian_qpole_fmm.hpp"

namespace fmm
{

// ---------------------------------------------------------------------------
// M2L device function — same math as CPU M2L, but with atomicAdd
// ---------------------------------------------------------------------------

template<class T, class Tacc>
__device__ void M2LGpu(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
                       const CartesianMultipole<T>& multipole,
                       CartesianLocalExpansion<Tacc>* local)
{
    T Rx = targetCenter[0] - sourceCenter[0];
    T Ry = targetCenter[1] - sourceCenter[1];
    T Rz = targetCenter[2] - sourceCenter[2];

    T r2       = Rx * Rx + Ry * Ry + Rz * Rz;
    T r_minus1 = ryoanji::inverseSquareRoot(r2);
    T r_minus2 = r_minus1 * r_minus1;
    T r_minus3 = r_minus2 * r_minus1;
    T r_minus5 = r_minus3 * r_minus2;
    T r_minus7 = r_minus5 * r_minus2;
    T r_minus9 = r_minus7 * r_minus2;

    T M = multipole[Cqi::mass];

    T QRx = multipole[Cqi::qxx] * Rx + multipole[Cqi::qxy] * Ry + multipole[Cqi::qxz] * Rz;
    T QRy = multipole[Cqi::qxy] * Rx + multipole[Cqi::qyy] * Ry + multipole[Cqi::qyz] * Rz;
    T QRz = multipole[Cqi::qxz] * Rx + multipole[Cqi::qyz] * Ry + multipole[Cqi::qzz] * Rz;

    T rQr = Rx * QRx + Ry * QRy + Rz * QRz;

    T mono5  = M * r_minus5;
    T rQr_r7 = rQr * r_minus7;
    T rQr_r9 = rQr * r_minus9;

    // 10 atomicAdd calls for the local expansion components (compute in T, accumulate in Tacc)
    atomicAdd(&((*local)[Cli::pot]), Tacc(M * r_minus1 + T(0.5) * rQr * r_minus5));

    atomicAdd(&((*local)[Cli::gx]), Tacc(-M * Rx * r_minus3 + QRx * r_minus5 - T(2.5) * rQr * Rx * r_minus7));
    atomicAdd(&((*local)[Cli::gy]), Tacc(-M * Ry * r_minus3 + QRy * r_minus5 - T(2.5) * rQr * Ry * r_minus7));
    atomicAdd(&((*local)[Cli::gz]), Tacc(-M * Rz * r_minus3 + QRz * r_minus5 - T(2.5) * rQr * Rz * r_minus7));

    atomicAdd(&((*local)[Cli::txx]), Tacc(mono5 * (T(3) * Rx * Rx - r2) + multipole[Cqi::qxx] * r_minus5 -
                                      T(10) * QRx * Rx * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Rx * Rx));

    atomicAdd(&((*local)[Cli::txy]), Tacc(mono5 * T(3) * Rx * Ry + multipole[Cqi::qxy] * r_minus5 -
                                      T(5) * (QRx * Ry + QRy * Rx) * r_minus7 + T(17.5) * rQr_r9 * Rx * Ry));

    atomicAdd(&((*local)[Cli::txz]), Tacc(mono5 * T(3) * Rx * Rz + multipole[Cqi::qxz] * r_minus5 -
                                      T(5) * (QRx * Rz + QRz * Rx) * r_minus7 + T(17.5) * rQr_r9 * Rx * Rz));

    atomicAdd(&((*local)[Cli::tyy]), Tacc(mono5 * (T(3) * Ry * Ry - r2) + multipole[Cqi::qyy] * r_minus5 -
                                      T(10) * QRy * Ry * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Ry * Ry));

    atomicAdd(&((*local)[Cli::tyz]), Tacc(mono5 * T(3) * Ry * Rz + multipole[Cqi::qyz] * r_minus5 -
                                      T(5) * (QRy * Rz + QRz * Ry) * r_minus7 + T(17.5) * rQr_r9 * Ry * Rz));

    atomicAdd(&((*local)[Cli::tzz]), Tacc(mono5 * (T(3) * Rz * Rz - r2) + multipole[Cqi::qzz] * r_minus5 -
                                      T(10) * QRz * Rz * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Rz * Rz));
}

// ---------------------------------------------------------------------------
// P2M kernel — compute leaf multipoles on GPU
// ---------------------------------------------------------------------------

template<int TPL, class T>
__global__ void computeLeafMultipolesGpuKernel(const T* x, const T* y, const T* z, const T* m,
                                                const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                                const LocalIndex* layout, const Vec4<T>* centers,
                                                CartesianMultipole<T>* multipoles)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;
    TreeNodeIndex internalIdx;

    CartesianMultipole<T> mp_loc;
    for (auto& v : mp_loc)
        v = T(0);

    if (leafIdx < numLeaves)
    {
        internalIdx = leafToInternal[leafIdx];
        auto com    = centers[internalIdx];
        P2M_add<TPL>(x, y, z, m, layout[leafIdx] + threadIdx.x % TPL, layout[leafIdx + 1], com, mp_loc);
    }

    constexpr int mpNumElements = 8;
#pragma unroll
    for (int offset = 1; offset < TPL; offset *= 2)
    {
#pragma unroll
        for (int mi = 0; mi < mpNumElements; ++mi)
            mp_loc[mi] = mp_loc[mi] + cstone::shflDownSync(mp_loc[mi], offset);
    }

    if (tid % TPL == 0 && leafIdx < numLeaves) { multipoles[internalIdx] = P2M_finalize(mp_loc); }
}

template<class T>
void computeLeafMultipolesGpu(const T* d_x, const T* d_y, const T* d_z, const T* d_m,
                               const TreeNodeIndex* d_leafToInternal, TreeNodeIndex numLeaves,
                               const LocalIndex* d_layout, const Vec4<T>* d_centers,
                               CartesianMultipole<T>* d_multipoles)
{
    constexpr int numThreads     = 256;
    constexpr int threadsPerLeaf = 8;
    int           numBlocks      = cstone::iceil(threadsPerLeaf * numLeaves, numThreads);

    if (numBlocks)
    {
        computeLeafMultipolesGpuKernel<threadsPerLeaf>
            <<<numBlocks, numThreads>>>(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers,
                                        d_multipoles);
    }
}

// ---------------------------------------------------------------------------
// M2M upsweep kernel
// ---------------------------------------------------------------------------

template<class T>
__global__ void upsweepMultipolesGpuKernel(TreeNodeIndex firstCell, TreeNodeIndex lastCell,
                                            const TreeNodeIndex* childOffsets, const Vec4<T>* centers,
                                            CartesianMultipole<T>* multipoles)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int     cellIdx = tid / 8 + firstCell;

    TreeNodeIndex firstChild = 0;
    if (cellIdx < lastCell) { firstChild = childOffsets[cellIdx]; }

    CartesianMultipole<T> Mout;
    for (auto& v : Mout)
        v = T(0);

    if (firstChild)
    {
        int     child = firstChild + threadIdx.x % 8;
        Vec3<T> dX    = util::makeVec3(centers[cellIdx] - centers[child]);
        addQuadrupole(Mout, dX, multipoles[child]);
    }

    constexpr int mpNumElements = 8;
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
                           const Vec4<T>* d_centers, CartesianMultipole<T>* d_multipoles)
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
                firstCell, lastCell, d_childOffsets, d_centers, d_multipoles);
        }
    }
}

// ---------------------------------------------------------------------------
// L2L downsweep kernel
// ---------------------------------------------------------------------------

template<class T, class Tacc>
__global__ void l2lKernel(TreeNodeIndex start, TreeNodeIndex end, const TreeNodeIndex* childOffsets,
                          const Vec4<T>* centers, CartesianLocalExpansion<Tacc>* locals)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end) return;

    TreeNodeIndex firstChild = childOffsets[i];
    if (firstChild == 0) return; // leaf node

    L2L(firstChild, firstChild + 8, centers[i], centers, locals[i], locals);
}

template<class T, class Tacc>
void downsweepLocalExpansionsGpu(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* d_childOffsets,
                                  const Vec4<T>* d_centers, CartesianLocalExpansion<Tacc>* d_locals)
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
                                                                              d_locals);
        }
    }
}

// ---------------------------------------------------------------------------
// L2P kernel
// ---------------------------------------------------------------------------

template<class T, class Tacc>
__global__ void l2pKernel(TreeNodeIndex firstLeaf, TreeNodeIndex lastLeaf, const TreeNodeIndex* leafToInternalMap,
                          const LocalIndex* layout, const Vec4<T>* centers,
                          const CartesianLocalExpansion<Tacc>* locals, const T* x, const T* y, const T* z,
                          Tacc* ppot, Tacc* pax, Tacc* pay, Tacc* paz, LocalIndex firstTarget)
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
        LocalIndex  ti = t - firstTarget;
        Vec4<Tacc>  acc{ppot[ti], pax[ti], pay[ti], paz[ti]};
        Vec3<T>     target{x[t], y[t], z[t]};
        acc      = L2P(acc, target, center, L);
        ppot[ti] = acc[0];
        pax[ti]  = acc[1];
        pay[ti]  = acc[2];
        paz[ti]  = acc[3];
    }
}

// ---------------------------------------------------------------------------
// Dual traversal kernel
// ---------------------------------------------------------------------------

using CartTravConfig = cstone::TraversalConfig<1024, 224, 704, 512, 160, 96, 192, 640, 224, 32>;

struct CartDualConfig
{
    static constexpr unsigned numWarps           = 7;
    static constexpr unsigned numThreadsPerBlock = numWarps * cstone::GpuConfig::warpSize;
    static constexpr unsigned kBlocksPerCluster  = 8;
};

template<int numWarps, class T, class Tacc>
__global__ void cartFmmDualTraversalKernel(const TreeNodeIndex* __restrict__ childOffsets,
                                           const Vec3<T>* __restrict__ geoSizes,
                                           const Vec4<T>* __restrict__ centers,
                                           const CartesianMultipole<T>* __restrict__ multipoles,
                                           CartesianLocalExpansion<Tacc>* __restrict__ locals,
                                           const TreeNodeIndex* __restrict__ internalToLeaf,
                                           const LocalIndex* __restrict__ layout,
                                           const T* __restrict__ x, const T* __restrict__ y,
                                           const T* __restrict__ z, const T* __restrict__ h, const T* __restrict__ m,
                                           Tacc* __restrict__ ppot, Tacc* __restrict__ pax, Tacc* __restrict__ pay,
                                           Tacc* __restrict__ paz, LocalIndex firstTarget,
                                           T invTheta,
                                           cstone::GlobalWorkQueue gq, cstone::GlobalTraversalQueue tq,
                                           unsigned* nProd,
                                           unsigned* d_m2lCount, unsigned* d_p2pCount)
{
    // Opening angle MAC: l/r < theta  =>  r² > (l/theta)²
    auto continuation = [geoSizes, centers, invTheta] __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> comA = util::makeVec3(centers[a]);
        Vec3<T> comB = util::makeVec3(centers[b]);
        T       dist2 = norm2(comA - comB);

        Vec3<T> sizeA = geoSizes[a];
        Vec3<T> sizeB = geoSizes[b];

        T lA        = T(2) * max(max(sizeA[0], sizeA[1]), sizeA[2]);
        T lB        = T(2) * max(max(sizeB[0], sizeB[1]), sizeB[2]);
        T threshold = max(lA, lB) * invTheta;

        return dist2 < threshold * threshold;
    };

    auto m2l = [centers, multipoles, locals, d_m2lCount] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        atomicAdd(d_m2lCount, 1u);
        M2LGpu(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], &locals[a]);
    };

    auto p2p = [internalToLeaf, layout, x, y, z, h, m, ppot, pax, pay, paz,
                firstTarget, d_p2pCount] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        atomicAdd(d_p2pCount, 1u);
        TreeNodeIndex aLeaf = internalToLeaf[a];
        TreeNodeIndex bLeaf = internalToLeaf[b];

        LocalIndex aFirst = layout[aLeaf];
        LocalIndex aLast  = layout[aLeaf + 1];
        LocalIndex bFirst = layout[bLeaf];
        LocalIndex bLast  = layout[bLeaf + 1];

        for (LocalIndex t = aFirst; t < aLast; ++t)
        {
            Vec4<Tacc> acc{0, 0, 0, 0};
            Vec3<T>    target{x[t], y[t], z[t]};
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

    cstone::dualTraversalGPU<numWarps, CartTravConfig>(childOffsets, TreeNodeIndex(0), TreeNodeIndex(0), gq, tq, nProd,
                                                       continuation, m2l, p2p);
}

// ---------------------------------------------------------------------------
// Host orchestration with CUDA event timing
// ---------------------------------------------------------------------------

template<class T, class KeyType>
void computeGravityFMMGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                           const TreeNodeIndex* internalToLeaf,
                           std::span<const TreeNodeIndex> leafToInternalMap,
                           std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                           const CartesianMultipole<T>* multipoles, const LocalIndex* layout,
                           TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                           const T* z, const T* h, const T* m, const cstone::Box<T>& box, float theta, float G,
                           T* ugrav, T* ax, T* ay, T* az, T* ugravTot, LocalIndex numParticles)
{
    using Tacc = float;

    TreeNodeIndex numNodes    = levelRange.back();
    TreeNodeIndex numLeaves   = TreeNodeIndex(leafToInternalMap.size());
    T             invTheta    = T(1) / T(theta);
    LocalIndex    firstTarget = layout[firstLeafIndex];
    LocalIndex    lastTarget  = layout[lastLeafIndex];
    LocalIndex    numTargets  = lastTarget - firstTarget;

    // 1. Compute geometric sizes from SFC prefixes
    std::vector<Vec3<T>> h_geoSizes(numNodes);
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        KeyType  prefix   = prefixes[i];
        KeyType  startKey = cstone::decodePlaceholderBit(prefix);
        unsigned level    = cstone::decodePrefixLength(prefix) / 3;
        auto     nodeBox  = cstone::sfcIBox(cstone::sfcKey(startKey), level);
        auto [center, sz] = cstone::centerAndSize<KeyType>(nodeBox, box);
        h_geoSizes[i]     = sz;
    }

    // 2. Upload tree structure arrays
    Vec3<T>*       d_geoSizes;
    TreeNodeIndex* d_childOffsets;
    TreeNodeIndex* d_internalToLeaf;
    TreeNodeIndex* d_leafToInternal;
    LocalIndex*    d_layout;
    Vec4<T>*       d_centers;

    checkGpuErrors(cudaMalloc(&d_geoSizes, numNodes * sizeof(Vec3<T>)));
    checkGpuErrors(cudaMalloc(&d_childOffsets, (numNodes + 1) * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_internalToLeaf, numNodes * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_leafToInternal, numLeaves * sizeof(TreeNodeIndex)));
    checkGpuErrors(cudaMalloc(&d_layout, (numLeaves + 1) * sizeof(LocalIndex)));
    checkGpuErrors(cudaMalloc(&d_centers, numNodes * sizeof(Vec4<T>)));

    checkGpuErrors(cudaMemcpy(d_geoSizes, h_geoSizes.data(), numNodes * sizeof(Vec3<T>), cudaMemcpyHostToDevice));
    checkGpuErrors(
        cudaMemcpy(d_childOffsets, childOffsets, (numNodes + 1) * sizeof(TreeNodeIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(
        cudaMemcpy(d_internalToLeaf, internalToLeaf, numNodes * sizeof(TreeNodeIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_leafToInternal, leafToInternalMap.data(), numLeaves * sizeof(TreeNodeIndex),
                              cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_layout, layout, (numLeaves + 1) * sizeof(LocalIndex), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_centers, centers, numNodes * sizeof(Vec4<T>), cudaMemcpyHostToDevice));

    // 3. Upload particle arrays
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

    // 4. Create CUDA events for phase timing
    cudaEvent_t evUpsweepStart, evUpsweepEnd, evTravStart, evTravEnd;
    cudaEvent_t evL2LStart, evL2LEnd, evL2PStart, evL2PEnd;
    checkGpuErrors(cudaEventCreate(&evUpsweepStart));
    checkGpuErrors(cudaEventCreate(&evUpsweepEnd));
    checkGpuErrors(cudaEventCreate(&evTravStart));
    checkGpuErrors(cudaEventCreate(&evTravEnd));
    checkGpuErrors(cudaEventCreate(&evL2LStart));
    checkGpuErrors(cudaEventCreate(&evL2LEnd));
    checkGpuErrors(cudaEventCreate(&evL2PStart));
    checkGpuErrors(cudaEventCreate(&evL2PEnd));

    // 5. Allocate device multipoles and compute P2M + M2M on GPU
    CartesianMultipole<T>* d_multipoles;
    checkGpuErrors(cudaMalloc(&d_multipoles, numNodes * sizeof(CartesianMultipole<T>)));
    checkGpuErrors(cudaMemset(d_multipoles, 0, numNodes * sizeof(CartesianMultipole<T>)));

    checkGpuErrors(cudaEventRecord(evUpsweepStart));

    computeLeafMultipolesGpu(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers, d_multipoles);
    upsweepMultipolesGpu(levelRange, d_childOffsets, d_centers, d_multipoles);

    checkGpuErrors(cudaEventRecord(evUpsweepEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 6. Allocate + zero-init device locals and particle accumulators (float for fast atomicAdd)
    CartesianLocalExpansion<Tacc>* d_locals;
    checkGpuErrors(cudaMalloc(&d_locals, numNodes * sizeof(CartesianLocalExpansion<Tacc>)));
    checkGpuErrors(cudaMemset(d_locals, 0, numNodes * sizeof(CartesianLocalExpansion<Tacc>)));

    Tacc *d_ppot, *d_pax, *d_pay, *d_paz;
    checkGpuErrors(cudaMalloc(&d_ppot, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMalloc(&d_pax, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMalloc(&d_pay, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMalloc(&d_paz, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMemset(d_ppot, 0, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMemset(d_pax, 0, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMemset(d_pay, 0, numTargets * sizeof(Tacc)));
    checkGpuErrors(cudaMemset(d_paz, 0, numTargets * sizeof(Tacc)));

    // 7. Allocate global work queues for dual traversal
    constexpr unsigned gChunk = CartTravConfig::chunkSize;
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

    constexpr unsigned tChunk = CartTravConfig::travChunkSize;
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
    constexpr unsigned kBlocksPerCluster = CartDualConfig::kBlocksPerCluster;
    constexpr unsigned numWarps          = CartDualConfig::numWarps;
    constexpr unsigned threadsPerBlock   = CartDualConfig::numThreadsPerBlock;

    unsigned maxBlocks = cstone::maxConcurrentBlocks(
        cartFmmDualTraversalKernel<numWarps, T, Tacc>, threadsPerBlock, kBlocksPerCluster);
    unsigned totalBlocks = maxBlocks;

    printf("[CartesianFMM GPU] Launching %u blocks (%u clusters of %u)\n",
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

    // 10. Allocate interaction counters
    unsigned* d_m2lCount;
    unsigned* d_p2pCount;
    checkGpuErrors(cudaMalloc(&d_m2lCount, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_p2pCount, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_m2lCount, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_p2pCount, 0, sizeof(unsigned)));

    // 11. Launch dual traversal
    checkGpuErrors(cudaEventRecord(evTravStart));

    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, cartFmmDualTraversalKernel<numWarps, T, Tacc>, d_childOffsets,
                                      d_geoSizes, d_centers, d_multipoles, d_locals, d_internalToLeaf, d_layout, d_x,
                                      d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget, invTheta,
                                      gq, tq, d_nProd, d_m2lCount, d_p2pCount));

    checkGpuErrors(cudaEventRecord(evTravEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // Print interaction counts
    unsigned h_m2l, h_p2p;
    checkGpuErrors(cudaMemcpy(&h_m2l, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(&h_p2p, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost));

    // 12. L2L downsweep
    checkGpuErrors(cudaEventRecord(evL2LStart));
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals);
    checkGpuErrors(cudaEventRecord(evL2LEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 13. L2P
    checkGpuErrors(cudaEventRecord(evL2PStart));
    {
        constexpr int numThreadsL2P = 256;
        TreeNodeIndex numLeafRange  = lastLeafIndex - firstLeafIndex;
        if (numLeafRange > 0)
        {
            l2pKernel<<<(numLeafRange + numThreadsL2P - 1) / numThreadsL2P, numThreadsL2P>>>(
                firstLeafIndex, lastLeafIndex, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z, d_ppot,
                d_pax, d_pay, d_paz, firstTarget);
        }
    }
    checkGpuErrors(cudaEventRecord(evL2PEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 14. Print timing breakdown
    float msUpsweep, msTrav, msL2L, msL2P;
    checkGpuErrors(cudaEventElapsedTime(&msUpsweep, evUpsweepStart, evUpsweepEnd));
    checkGpuErrors(cudaEventElapsedTime(&msTrav, evTravStart, evTravEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2L, evL2LStart, evL2LEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2P, evL2PStart, evL2PEnd));
    float msTotal = msUpsweep + msTrav + msL2L + msL2P;

    printf("[CartesianFMM GPU] P2M+M2M: %.3f ms | Traversal: %.3f ms | L2L: %.3f ms | L2P: %.3f ms | Total: %.3f ms\n",
           msUpsweep, msTrav, msL2L, msL2P, msTotal);
    printf("[CartesianFMM GPU] M2L calls: %u, P2P calls: %u\n", h_m2l, h_p2p);

    checkGpuErrors(cudaEventDestroy(evUpsweepStart));
    checkGpuErrors(cudaEventDestroy(evUpsweepEnd));
    checkGpuErrors(cudaEventDestroy(evTravStart));
    checkGpuErrors(cudaEventDestroy(evTravEnd));
    checkGpuErrors(cudaEventDestroy(evL2LStart));
    checkGpuErrors(cudaEventDestroy(evL2LEnd));
    checkGpuErrors(cudaEventDestroy(evL2PStart));
    checkGpuErrors(cudaEventDestroy(evL2PEnd));

    // 15. Download results (Tacc = float, G-scaling promotes to double)
    std::vector<Tacc> h_ppot(numTargets), h_pax(numTargets), h_pay(numTargets), h_paz(numTargets);
    checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pax.data(), d_pax, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pay.data(), d_pay, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_paz.data(), d_paz, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));

    // 16. Apply G scaling and accumulate into output arrays
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

    // 17. Free device memory
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

    cudaFree(d_m2lCount);
    cudaFree(d_p2pCount);
}

} // namespace fmm
