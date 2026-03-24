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
#include "cstone/focus/source_center_gpu.h"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
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
// M2L device function — FMA-optimized, register-local output, templatized on compute precision
// ---------------------------------------------------------------------------

/*! @brief Compute M2L contribution into a register-local expansion (no atomicAdd)
 *
 * @tparam Tc   Compute precision (float for mixed-precision, T for full precision)
 * @tparam T    Storage precision of positions/multipoles (typically double)
 * @tparam Tacc Accumulator precision of the local expansion (typically float)
 *
 * Displacement R is computed in T (full precision) then truncated to Tc,
 * avoiding catastrophic cancellation. All subsequent math uses Tc.
 * Explicit fma() calls guide nvcc to emit fused multiply-add instructions.
 */
template<class Tc, class T, class Tacc>
__device__ void M2LGpuCompute(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
                               const CartesianMultipole<T>& multipole,
                               CartesianLocalExpansion<Tacc>& local)
{
    Tc Rx = Tc(targetCenter[0] - sourceCenter[0]);
    Tc Ry = Tc(targetCenter[1] - sourceCenter[1]);
    Tc Rz = Tc(targetCenter[2] - sourceCenter[2]);

    Tc r2       = fma(Rx, Rx, fma(Ry, Ry, Rz * Rz));
    Tc r_minus1 = ryoanji::inverseSquareRoot(r2);
    Tc r_minus2 = r_minus1 * r_minus1;
    Tc r_minus3 = r_minus2 * r_minus1;
    Tc r_minus5 = r_minus3 * r_minus2;
    Tc r_minus7 = r_minus5 * r_minus2;
    Tc r_minus9 = r_minus7 * r_minus2;

    Tc M = Tc(multipole[Cqi::mass]);

    Tc qxx = Tc(multipole[Cqi::qxx]), qxy = Tc(multipole[Cqi::qxy]), qxz = Tc(multipole[Cqi::qxz]);
    Tc qyy = Tc(multipole[Cqi::qyy]), qyz = Tc(multipole[Cqi::qyz]), qzz = Tc(multipole[Cqi::qzz]);

    Tc QRx = fma(qxx, Rx, fma(qxy, Ry, qxz * Rz));
    Tc QRy = fma(qxy, Rx, fma(qyy, Ry, qyz * Rz));
    Tc QRz = fma(qxz, Rx, fma(qyz, Ry, qzz * Rz));

    Tc rQr = fma(Rx, QRx, fma(Ry, QRy, Rz * QRz));

    Tc mono5  = M * r_minus5;
    Tc rQr_r7 = rQr * r_minus7;
    Tc rQr_r9 = rQr * r_minus9;

    // L0: potential
    local[Cli::pot] += Tacc(fma(Tc(0.5) * rQr, r_minus5, M * r_minus1));

    // Li: gradient — factor R_i out of all three terms
    Tc gcoeff = fma(Tc(-2.5), rQr_r7, -M * r_minus3);
    local[Cli::gx] += Tacc(fma(gcoeff, Rx, QRx * r_minus5));
    local[Cli::gy] += Tacc(fma(gcoeff, Ry, QRy * r_minus5));
    local[Cli::gz] += Tacc(fma(gcoeff, Rz, QRz * r_minus5));

    // Lij: tidal tensor — precompute shared coefficients
    Tc RR_coeff   = fma(Tc(3), mono5, Tc(17.5) * rQr_r9);
    Tc diag_const = fma(-mono5, r2, Tc(-2.5) * rQr_r7);
    Tc neg5_r7    = Tc(-5) * r_minus7;
    Tc neg10_r7   = Tc(-10) * r_minus7;

    // Diagonal: tii = RR_coeff * Ri^2 + diag_const + qii/r^5 - 10*QRi*Ri/r^7
    local[Cli::txx] += Tacc(fma(RR_coeff, Rx * Rx, diag_const) +
                            fma(neg10_r7 * QRx, Rx, qxx * r_minus5));
    local[Cli::tyy] += Tacc(fma(RR_coeff, Ry * Ry, diag_const) +
                            fma(neg10_r7 * QRy, Ry, qyy * r_minus5));
    local[Cli::tzz] += Tacc(fma(RR_coeff, Rz * Rz, diag_const) +
                            fma(neg10_r7 * QRz, Rz, qzz * r_minus5));

    // Off-diagonal: tij = RR_coeff * Ri*Rj + qij/r^5 - 5*(QRi*Rj + QRj*Ri)/r^7
    local[Cli::txy] += Tacc(fma(RR_coeff, Rx * Ry,
                                fma(neg5_r7, fma(QRx, Ry, QRy * Rx), qxy * r_minus5)));
    local[Cli::txz] += Tacc(fma(RR_coeff, Rx * Rz,
                                fma(neg5_r7, fma(QRx, Rz, QRz * Rx), qxz * r_minus5)));
    local[Cli::tyz] += Tacc(fma(RR_coeff, Ry * Rz,
                                fma(neg5_r7, fma(QRy, Rz, QRz * Ry), qyz * r_minus5)));
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

using CartTravConfig = cstone::TraversalConfig<1024, 256, 768, 576, 192, 96, 192, 576, 224, 32, 1>;

struct CartDualConfig
{
    static constexpr unsigned numWarps           = 4;
    static constexpr unsigned numThreadsPerBlock = numWarps * cstone::GpuConfig::warpSize;
    static constexpr unsigned kBlocksPerCluster  = 8;
};

template<MacVariant macType, int numWarps, class T, class Tacc>
__global__ void cartFmmDualTraversalKernel(const TreeNodeIndex* __restrict__ childOffsets,
                                           const Vec3<T>* __restrict__ geoCenters,
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

    auto m2l = [centers, multipoles, locals] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        CartesianLocalExpansion<Tacc> partial{};
        M2LGpuCompute<float>(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], partial);
        for (int c = 0; c < 10; ++c)
            atomicAdd(&locals[a][c], partial[c]);
    };

    auto p2p = [internalToLeaf, layout, x, y, z, h, m, ppot, pax, pay, paz,
                firstTarget] __device__(unsigned p2pMask, TreeNodeIndex myA, TreeNodeIndex myB)
    {
        using cstone::shflSync;
        constexpr unsigned ws  = cstone::GpuConfig::warpSize;
        constexpr int      nwt = 2;

        unsigned lane     = threadIdx.x % ws;
        unsigned p2pCount = __popc(p2pMask);

        TreeNodeIndex prevA = ~TreeNodeIndex(0);
        LocalIndex    aFirst = 0, aLast = 0;
        Vec4<Tacc>    acc[nwt] = {};
        Vec3<T>       tpos[nwt];
        T             th[nwt];

        for (unsigned i = 0; i < p2pCount; ++i)
        {
            // Find lane holding the i-th P2P item
            unsigned tmp = p2pMask;
            for (unsigned skip = 0; skip < i; ++skip)
                tmp &= tmp - 1; // clear lowest set bit
            int srcLane = __ffs(tmp) - 1;

            TreeNodeIndex a = shflSync(myA, srcLane);
            TreeNodeIndex b = shflSync(myB, srcLane);

            // Target cell changed — flush accumulators and load new targets
            if (a != prevA)
            {
                if (prevA != ~TreeNodeIndex(0))
                {
                    for (int k = 0; k < nwt; ++k)
                    {
                        LocalIndex t = aFirst + k * ws + lane;
                        if (t < aLast)
                        {
                            LocalIndex ti = t - firstTarget;
                            atomicAdd(&ppot[ti], acc[k][0]);
                            atomicAdd(&pax[ti], acc[k][1]);
                            atomicAdd(&pay[ti], acc[k][2]);
                            atomicAdd(&paz[ti], acc[k][3]);
                        }
                    }
                }

                prevA = a;
                TreeNodeIndex aLeaf = internalToLeaf[a];
                aFirst = layout[aLeaf];
                aLast  = layout[aLeaf + 1];
                for (int k = 0; k < nwt; ++k)
                {
                    LocalIndex t = aFirst + k * ws + lane;
                    t            = min(t, aLast - 1);
                    tpos[k]      = {x[t], y[t], z[t]};
                    th[k]        = h[t];
                    acc[k]       = {0, 0, 0, 0};
                }
            }

            // Source cell b: broadcast sources across warp (BH directAcc pattern)
            TreeNodeIndex bLeaf  = internalToLeaf[b];
            LocalIndex    bFirst = layout[bLeaf];
            LocalIndex    bLast  = layout[bLeaf + 1];

            for (LocalIndex sBase = bFirst; sBase < bLast; sBase += ws)
            {
                LocalIndex s  = sBase + lane;
                T          sx = (s < bLast) ? x[s] : T(0);
                T          sy = (s < bLast) ? y[s] : T(0);
                T          sz = (s < bLast) ? z[s] : T(0);
                T          sm = (s < bLast) ? m[s] : T(0);
                T          sh = (s < bLast) ? h[s] : T(0);

                int count = min(ws, (unsigned)(bLast - sBase));
                for (int j = 0; j < count; ++j)
                {
                    Vec3<T> sj = {shflSync(sx, j), shflSync(sy, j), shflSync(sz, j)};
                    T       mj = shflSync(sm, j);
                    T       hj = shflSync(sh, j);

                    for (int k = 0; k < nwt; ++k)
                        acc[k] = ryoanji::P2P(acc[k], tpos[k], sj, mj, th[k], hj);
                }
            }
        }

        // Flush final target accumulators
        if (prevA != ~TreeNodeIndex(0))
        {
            for (int k = 0; k < nwt; ++k)
            {
                LocalIndex t = aFirst + k * ws + lane;
                if (t < aLast)
                {
                    LocalIndex ti = t - firstTarget;
                    atomicAdd(&ppot[ti], acc[k][0]);
                    atomicAdd(&pax[ti], acc[k][1]);
                    atomicAdd(&pay[ti], acc[k][2]);
                    atomicAdd(&paz[ti], acc[k][3]);
                }
            }
        }
    };

    cstone::dualTraversalGPUStatic<numWarps, CartTravConfig>(childOffsets, TreeNodeIndex(0), TreeNodeIndex(0), gq, tq,
                                                             nProd, continuation, m2l, p2p);
}

// ---------------------------------------------------------------------------
// Host orchestration with CUDA event timing
// ---------------------------------------------------------------------------

struct FmmGpuStats
{
    float msUpsweep{}, msTraversal{}, msL2L{}, msL2P{};
    float msTotal() const { return msUpsweep + msTraversal + msL2L + msL2P; }
};

template<MacVariant macType = ScalarMac, class T, class KeyType>
void computeGravityFMMGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                           const TreeNodeIndex* internalToLeaf,
                           std::span<const TreeNodeIndex> leafToInternalMap,
                           std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                           const CartesianMultipole<T>* multipoles, const LocalIndex* layout,
                           TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                           const T* z, const T* h, const T* m, const cstone::Box<T>& box, float G,
                           float invTheta, T* ugrav, T* ax, T* ay, T* az, T* ugravTot,
                           LocalIndex numParticles,
                           FmmGpuStats* stats = nullptr)
{
    using Tacc = float;

    TreeNodeIndex numNodes    = levelRange.back();
    TreeNodeIndex numLeaves   = TreeNodeIndex(leafToInternalMap.size());
    LocalIndex    firstTarget = layout[firstLeafIndex];
    LocalIndex    lastTarget  = layout[lastLeafIndex];
    LocalIndex    numTargets  = lastTarget - firstTarget;

    // 1. Compute geometric centers/sizes from SFC prefixes
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

    // 2. Upload tree structure arrays
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

    unsigned totalBlocks = kBlocksPerCluster * 64u;

    // Block/cluster count available via totalBlocks / kBlocksPerCluster

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
    checkGpuErrors(cudaEventRecord(evTravStart));

    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, cartFmmDualTraversalKernel<macType, numWarps, T, Tacc>,
                                      d_childOffsets,
                                      d_geoCenters, d_geoSizes,
                                      d_centers, d_multipoles, d_locals, d_internalToLeaf, d_layout, d_x,
                                      d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget,
                                      gq, tq, d_nProd));

    checkGpuErrors(cudaEventRecord(evTravEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 11. L2L downsweep
    checkGpuErrors(cudaEventRecord(evL2LStart));
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals);
    checkGpuErrors(cudaEventRecord(evL2LEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 12. L2P
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

    // 13. Print timing breakdown
    float msUpsweep, msTrav, msL2L, msL2P;
    checkGpuErrors(cudaEventElapsedTime(&msUpsweep, evUpsweepStart, evUpsweepEnd));
    checkGpuErrors(cudaEventElapsedTime(&msTrav, evTravStart, evTravEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2L, evL2LStart, evL2LEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2P, evL2PStart, evL2PEnd));
    float msTotal = msUpsweep + msTrav + msL2L + msL2P;

    // Timing available via FmmGpuStats output parameter

    if (stats) { *stats = {msUpsweep, msTrav, msL2L, msL2P}; }

    checkGpuErrors(cudaEventDestroy(evUpsweepStart));
    checkGpuErrors(cudaEventDestroy(evUpsweepEnd));
    checkGpuErrors(cudaEventDestroy(evTravStart));
    checkGpuErrors(cudaEventDestroy(evTravEnd));
    checkGpuErrors(cudaEventDestroy(evL2LStart));
    checkGpuErrors(cudaEventDestroy(evL2LEnd));
    checkGpuErrors(cudaEventDestroy(evL2PStart));
    checkGpuErrors(cudaEventDestroy(evL2PEnd));

    // 14. Download results (Tacc = float, G-scaling promotes to double)
    std::vector<Tacc> h_ppot(numTargets), h_pax(numTargets), h_pay(numTargets), h_paz(numTargets);
    checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pax.data(), d_pax, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_pay.data(), d_pay, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(h_paz.data(), d_paz, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));

    // 15. Apply G scaling and accumulate into output arrays
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

    // 16. Free device memory
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
// Helper kernels for GPU-native path
// ---------------------------------------------------------------------------

template<class T>
__global__ void scaleVec3Kernel(Vec3<T>* data, TreeNodeIndex n, T factor)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i][0] *= factor;
        data[i][1] *= factor;
        data[i][2] *= factor;
    }
}

template<class T, class Tacc>
__global__ void applyGScalingKernel(LocalIndex n, float G,
                                     const Tacc* pax, const Tacc* pay, const Tacc* paz,
                                     T* ax, T* ay, T* az)
{
    LocalIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        ax[i] += T(G) * T(pax[i]);
        ay[i] += T(G) * T(pay[i]);
        az[i] += T(G) * T(paz[i]);
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
    T* ugravTot, LocalIndex numParticles,
    FmmGpuStats* stats)
{
    using Tacc = float;

    LocalIndex firstTarget = 0;
    LocalIndex numTargets  = numParticles;

    // 1. Compute geometric centers/sizes on GPU from SFC prefixes
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
        if (nb) { scaleVec3Kernel<<<nb, nt>>>(d_geoSizes, numNodes, T(invTheta)); }
    }

    // 2. Create CUDA events for phase timing
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

    // 3. Allocate device multipoles and compute P2M + M2M on GPU
    CartesianMultipole<T>* d_multipoles;
    checkGpuErrors(cudaMalloc(&d_multipoles, numNodes * sizeof(CartesianMultipole<T>)));
    checkGpuErrors(cudaMemset(d_multipoles, 0, numNodes * sizeof(CartesianMultipole<T>)));

    checkGpuErrors(cudaEventRecord(evUpsweepStart));

    computeLeafMultipolesGpu(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers, d_multipoles);
    upsweepMultipolesGpu(levelRange, d_childOffsets, d_centers, d_multipoles);

    checkGpuErrors(cudaEventRecord(evUpsweepEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 4. Allocate + zero-init device locals and particle accumulators
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

    // 5. Allocate global work queues for dual traversal
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

    // 6. Reset queues
    checkGpuErrors(cudaMemset(d_wHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_rHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segR, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_twHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_trHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned)));

    // 7. Configure cluster launch
    constexpr unsigned kBlocksPerCluster = CartDualConfig::kBlocksPerCluster;
    constexpr unsigned numWarps          = CartDualConfig::numWarps;
    constexpr unsigned threadsPerBlock   = CartDualConfig::numThreadsPerBlock;

    unsigned totalBlocks = kBlocksPerCluster * 64u;

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
    checkGpuErrors(cudaEventRecord(evTravStart));

    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, cartFmmDualTraversalKernel<macType, numWarps, T, Tacc>,
                                      d_childOffsets,
                                      d_geoCenters, d_geoSizes,
                                      d_centers, d_multipoles, d_locals, d_internalToLeaf, d_layout, d_x,
                                      d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget,
                                      gq, tq, d_nProd));

    checkGpuErrors(cudaEventRecord(evTravEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 9. L2L downsweep
    checkGpuErrors(cudaEventRecord(evL2LStart));
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals);
    checkGpuErrors(cudaEventRecord(evL2LEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 10. L2P
    checkGpuErrors(cudaEventRecord(evL2PStart));
    {
        constexpr int numThreadsL2P = 256;
        if (numLeaves > 0)
        {
            l2pKernel<<<(numLeaves + numThreadsL2P - 1) / numThreadsL2P, numThreadsL2P>>>(
                0, numLeaves, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z, d_ppot,
                d_pax, d_pay, d_paz, firstTarget);
        }
    }
    checkGpuErrors(cudaEventRecord(evL2PEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 11. Timing
    float msUpsweep, msTrav, msL2L, msL2P;
    checkGpuErrors(cudaEventElapsedTime(&msUpsweep, evUpsweepStart, evUpsweepEnd));
    checkGpuErrors(cudaEventElapsedTime(&msTrav, evTravStart, evTravEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2L, evL2LStart, evL2LEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2P, evL2PStart, evL2PEnd));

    if (stats) { *stats = {msUpsweep, msTrav, msL2L, msL2P}; }

    checkGpuErrors(cudaEventDestroy(evUpsweepStart));
    checkGpuErrors(cudaEventDestroy(evUpsweepEnd));
    checkGpuErrors(cudaEventDestroy(evTravStart));
    checkGpuErrors(cudaEventDestroy(evTravEnd));
    checkGpuErrors(cudaEventDestroy(evL2LStart));
    checkGpuErrors(cudaEventDestroy(evL2LEnd));
    checkGpuErrors(cudaEventDestroy(evL2PStart));
    checkGpuErrors(cudaEventDestroy(evL2PEnd));

    // 12. Apply G-scaling on GPU and accumulate into caller's output buffers
    {
        int nt = 256;
        int nb = cstone::iceil(numTargets, nt);
        if (nb)
        {
            applyGScalingKernel<<<nb, nt>>>(numTargets, G, d_pax, d_pay, d_paz,
                                            d_ax, d_ay, d_az);
        }
    }

    // 13. Download potential sum for ugravTot (small scalar)
    if (ugravTot)
    {
        // Compute ugrav = sum(G * m[i] * ppot[i]) / 2 on host
        std::vector<Tacc> h_ppot(numTargets);
        std::vector<T>    h_m(numTargets);
        checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot, numTargets * sizeof(Tacc), cudaMemcpyDeviceToHost));
        checkGpuErrors(cudaMemcpy(h_m.data(), d_m, numTargets * sizeof(T), cudaMemcpyDeviceToHost));
        T ugravLoc = 0;
        for (LocalIndex i = 0; i < numTargets; ++i)
            ugravLoc += G * h_m[i] * h_ppot[i];
        *ugravTot += T(0.5) * ugravLoc;
    }

    checkGpuErrors(cudaDeviceSynchronize());

    // 14. Free internally-allocated temporaries (caller owns input device pointers)
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

// ---------------------------------------------------------------------------
// Distributed overload — accepts pre-computed multipoles from MultipoleHolder::upsweep()
// ---------------------------------------------------------------------------

/*! @brief Run FMM dual traversal using multipoles computed by a distributed upsweep
 *
 * @param d_prefixes         SFC node keys on GPU (from octreeViewAcc().prefixes)
 * @param d_childOffsets     child offset array on GPU
 * @param d_internalToLeaf   internal-to-leaf map on GPU
 * @param d_leafToInternal   leaf-to-internal map on GPU (leaf portion only)
 * @param d_layout           particle offsets per leaf cell on GPU
 * @param d_centers          expansion centers on GPU (from expansionCentersAcc())
 * @param d_multipoles       pre-computed multipoles on GPU (from MultipoleHolder::upsweep())
 * @param levelRange         CPU-side level range span (from octreeViewAcc().levelRangeSpan())
 * @param numNodes           total number of octree nodes
 * @param numLeaves          number of leaf nodes
 * @param numParticlesWithHalos  total particles including halos
 * @param firstOwnedParticle    start of owned particle range
 * @param lastOwnedParticle     end of owned particle range
 * @param d_x,d_y,d_z,d_h,d_m  particle data on GPU
 * @param box                   coordinate bounding box
 * @param G                     gravitational constant
 * @param invTheta              1/theta for MAC
 * @param d_ax,d_ay,d_az        output accelerations on GPU (G-scaled, owned range only)
 * @param ugravTot              output potential sum (owned particles only)
 * @param stats                 optional timing stats (msUpsweep will be 0)
 */
template<MacVariant macType = ScalarMac, class T, class KeyType>
void computeGravityFMMGpuDistributed(
    const KeyType* d_prefixes,
    const TreeNodeIndex* d_childOffsets,
    const TreeNodeIndex* d_internalToLeaf,
    const TreeNodeIndex* d_leafToInternal,
    const LocalIndex* d_layout,
    const cstone::SourceCenterType<T>* d_centers,
    const CartesianMultipole<T>* d_multipoles,
    std::span<const TreeNodeIndex> levelRange,
    TreeNodeIndex numNodes, TreeNodeIndex numLeaves,
    LocalIndex numParticlesWithHalos,
    LocalIndex firstOwnedParticle, LocalIndex lastOwnedParticle,
    const T* d_x, const T* d_y, const T* d_z, const T* d_h, const T* d_m,
    const cstone::Box<T>& box, float G, float invTheta,
    T* d_ax, T* d_ay, T* d_az,
    T* ugravTot,
    FmmGpuStats* stats = nullptr)
{
    using Tacc = float;

    LocalIndex firstTarget = 0;
    LocalIndex numTargets  = numParticlesWithHalos;

    // 1. Compute geometric centers/sizes on GPU from SFC prefixes
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
        if (nb) { scaleVec3Kernel<<<nb, nt>>>(d_geoSizes, numNodes, T(invTheta)); }
    }

    // 2. Create CUDA events for phase timing
    cudaEvent_t evTravStart, evTravEnd;
    cudaEvent_t evL2LStart, evL2LEnd, evL2PStart, evL2PEnd;
    checkGpuErrors(cudaEventCreate(&evTravStart));
    checkGpuErrors(cudaEventCreate(&evTravEnd));
    checkGpuErrors(cudaEventCreate(&evL2LStart));
    checkGpuErrors(cudaEventCreate(&evL2LEnd));
    checkGpuErrors(cudaEventCreate(&evL2PStart));
    checkGpuErrors(cudaEventCreate(&evL2PEnd));

    // 3. Allocate + zero-init device locals and particle accumulators
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

    // 4. Allocate global work queues for dual traversal
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

    // 5. Reset queues
    checkGpuErrors(cudaMemset(d_wHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_rHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segCount, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_segR, 0, gSegs * sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_twHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_trHead, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_tsegR, 0, tSegs * sizeof(unsigned)));

    // 6. Configure cluster launch
    constexpr unsigned kBlocksPerCluster = CartDualConfig::kBlocksPerCluster;
    constexpr unsigned numWarps          = CartDualConfig::numWarps;
    constexpr unsigned threadsPerBlock   = CartDualConfig::numThreadsPerBlock;

    unsigned totalBlocks = kBlocksPerCluster * 64u;

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

    // 7. Launch dual traversal
    checkGpuErrors(cudaEventRecord(evTravStart));

    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, cartFmmDualTraversalKernel<macType, numWarps, T, Tacc>,
                                      d_childOffsets,
                                      d_geoCenters, d_geoSizes,
                                      d_centers, d_multipoles, d_locals, d_internalToLeaf, d_layout, d_x,
                                      d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget,
                                      gq, tq, d_nProd));

    checkGpuErrors(cudaEventRecord(evTravEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 8. L2L downsweep
    checkGpuErrors(cudaEventRecord(evL2LStart));
    downsweepLocalExpansionsGpu(levelRange, d_childOffsets, d_centers, d_locals);
    checkGpuErrors(cudaEventRecord(evL2LEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 9. L2P — all leaves
    checkGpuErrors(cudaEventRecord(evL2PStart));
    {
        constexpr int numThreadsL2P = 256;
        if (numLeaves > 0)
        {
            l2pKernel<<<(numLeaves + numThreadsL2P - 1) / numThreadsL2P, numThreadsL2P>>>(
                0, numLeaves, d_leafToInternal, d_layout, d_centers, d_locals, d_x, d_y, d_z, d_ppot,
                d_pax, d_pay, d_paz, firstTarget);
        }
    }
    checkGpuErrors(cudaEventRecord(evL2PEnd));
    checkGpuErrors(cudaDeviceSynchronize());

    // 10. Timing
    float msTrav, msL2L, msL2P;
    checkGpuErrors(cudaEventElapsedTime(&msTrav, evTravStart, evTravEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2L, evL2LStart, evL2LEnd));
    checkGpuErrors(cudaEventElapsedTime(&msL2P, evL2PStart, evL2PEnd));

    if (stats) { *stats = {0.0f, msTrav, msL2L, msL2P}; }

    checkGpuErrors(cudaEventDestroy(evTravStart));
    checkGpuErrors(cudaEventDestroy(evTravEnd));
    checkGpuErrors(cudaEventDestroy(evL2LStart));
    checkGpuErrors(cudaEventDestroy(evL2LEnd));
    checkGpuErrors(cudaEventDestroy(evL2PStart));
    checkGpuErrors(cudaEventDestroy(evL2PEnd));

    // 11. Apply G-scaling on owned range only
    {
        LocalIndex numOwned = lastOwnedParticle - firstOwnedParticle;
        int        nt       = 256;
        int        nb       = cstone::iceil(numOwned, nt);
        if (nb)
        {
            applyGScalingKernel<<<nb, nt>>>(numOwned, G,
                                            d_pax + firstOwnedParticle,
                                            d_pay + firstOwnedParticle,
                                            d_paz + firstOwnedParticle,
                                            d_ax + firstOwnedParticle,
                                            d_ay + firstOwnedParticle,
                                            d_az + firstOwnedParticle);
        }
    }

    // 12. Potential sum for owned particles only
    if (ugravTot)
    {
        LocalIndex    numOwned = lastOwnedParticle - firstOwnedParticle;
        std::vector<Tacc> h_ppot(numOwned);
        std::vector<T>    h_m(numOwned);
        checkGpuErrors(cudaMemcpy(h_ppot.data(), d_ppot + firstOwnedParticle, numOwned * sizeof(Tacc),
                                  cudaMemcpyDeviceToHost));
        checkGpuErrors(cudaMemcpy(h_m.data(), d_m + firstOwnedParticle, numOwned * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        T ugravLoc = 0;
        for (LocalIndex i = 0; i < numOwned; ++i)
            ugravLoc += G * h_m[i] * h_ppot[i];
        *ugravTot += T(0.5) * ugravLoc;
    }

    checkGpuErrors(cudaDeviceSynchronize());

    // 13. Free internally-allocated temporaries
    cudaFree(d_geoCenters);
    cudaFree(d_geoSizes);

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
