/*
 * Spherical harmonic multipole GPU kernels for FMM
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU port of spherical harmonic multipole FMM kernels
 *
 * Uses thrust::complex<T> instead of std::complex<T> for device compatibility.
 * Memory layout is identical, so CPU-computed data can be cudaMemcpy'd directly.
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
// GPU type aliases
// ---------------------------------------------------------------------------

template<class T>
using GpuComplex = thrust::complex<T>;

template<class T, int P = ExpansionOrder>
using GpuSphericalMultipole = util::array<GpuComplex<T>, Nterm<P>>;

template<class T, int P = ExpansionOrder>
using GpuSphericalLocalExpansion = util::array<GpuComplex<T>, Nterm<P>>;

// ---------------------------------------------------------------------------
// GPU tables
// ---------------------------------------------------------------------------

struct GpuSphericalTables
{
    double*            prefactor; // [4*P*P] on device
    double*            Anm;       // [4*P*P] on device
    GpuComplex<double>* Cnm;      // [P*P*P*P] on device
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
    checkGpuErrors(cudaMalloc(&gpu.Cnm, PP4 * sizeof(GpuComplex<double>)));

    checkGpuErrors(cudaMemcpy(gpu.prefactor, tab.prefactor.data(), 4 * PP2 * sizeof(double), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(gpu.Anm, tab.Anm.data(), 4 * PP2 * sizeof(double), cudaMemcpyHostToDevice));
    // std::complex<double> and thrust::complex<double> have identical layout
    checkGpuErrors(cudaMemcpy(gpu.Cnm, tab.Cnm.data(), PP4 * sizeof(GpuComplex<double>), cudaMemcpyHostToDevice));

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
// Device helpers
// ---------------------------------------------------------------------------

__device__ __host__ constexpr int oddevenGpu(int n) { return (n & 1) ? -1 : 1; }

template<class T>
__device__ void atomicAddComplex(GpuComplex<T>* addr, GpuComplex<T> val)
{
    atomicAdd(reinterpret_cast<T*>(addr), val.real());
    atomicAdd(reinterpret_cast<T*>(addr) + 1, val.imag());
}

// ---------------------------------------------------------------------------
// Device coordinate conversions
// ---------------------------------------------------------------------------

template<class T>
__device__ void cart2sphGpu(T& r, T& theta, T& phi, Vec3<T> dist)
{
    constexpr T EPS = T(1e-6);
    r               = sqrt(norm2(dist)) + EPS;
    theta           = acos(dist[2] / r);
    T absx          = abs(dist[0]);
    T absy          = abs(dist[1]);
    if (absx + absy < EPS) { phi = 0; }
    else if (absx < EPS) { phi = dist[1] / absy * T(M_PI) * T(0.5); }
    else if (dist[0] > 0) { phi = atan(dist[1] / dist[0]); }
    else { phi = atan(dist[1] / dist[0]) + T(M_PI); }
}

template<class T>
__device__ Vec3<T> sph2cartGpu(T r, T theta, T phi, Vec3<T> spherical)
{
    T st = sin(theta);
    T ct = cos(theta);
    T sp = sin(phi);
    T cp = cos(phi);

    return {st * cp * spherical[0] + ct * cp / r * spherical[1] - sp / r / st * spherical[2],
            st * sp * spherical[0] + ct * sp / r * spherical[1] + cp / r / st * spherical[2],
            ct * spherical[0] - st / r * spherical[1]};
}

// ---------------------------------------------------------------------------
// Solid harmonics evaluation (device)
// ---------------------------------------------------------------------------

template<int P, class T>
__device__ void evalMultipoleGpu(GpuComplex<T>* Ynm, GpuComplex<T>* YnmTheta, const double* prefactor, T rho, T alpha,
                                 T beta)
{
    const GpuComplex<T> I(0, 1);
    T                   x    = cos(alpha);
    T                   y    = sin(alpha);
    T                   fact = 1;
    T                   pn   = 1;
    T                   rhom = 1;

    for (int m = 0; m != P; ++m)
    {
        GpuComplex<T> eim = thrust::exp(I * T(m * beta));
        T             p   = pn;
        int           npn = m * m + 2 * m;
        int           nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = thrust::conj(Ynm[npn]);

        T p1 = p;
        p    = x * (2 * m + 1) * p1;

        YnmTheta[npn] = rhom * (p - (m + 1) * x * p1) / y * T(prefactor[npn]) * eim;

        rhom *= rho;
        T rhon = rhom;

        for (int n = m + 1; n != P; ++n)
        {
            int npm = n * n + n + m;
            int nmm = n * n + n - m;

            Ynm[npm] = rhon * p * T(prefactor[npm]) * eim;
            Ynm[nmm] = thrust::conj(Ynm[npm]);

            T p2 = p1;
            p1   = p;
            p    = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);

            YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) / y * T(prefactor[npm]) * eim;
            rhon *= rho;
        }

        pn = -pn * fact * y;
        fact += 2;
    }
}

template<int P, class T>
__device__ void evalLocalGpu(GpuComplex<T>* Ynm, GpuComplex<T>* YnmTheta, const double* prefactor, T rho, T alpha,
                             T beta)
{
    const GpuComplex<T> I(0, 1);
    T                   x    = cos(alpha);
    T                   y    = sin(alpha);
    T                   fact = 1;
    T                   pn   = 1;
    T                   rhom = T(1) / rho;

    for (int m = 0; m != 2 * P; ++m)
    {
        GpuComplex<T> eim = thrust::exp(I * T(m * beta));
        T             p   = pn;
        int           npn = m * m + 2 * m;
        int           nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = thrust::conj(Ynm[npn]);

        T p1 = p;
        p    = x * (2 * m + 1) * p1;

        YnmTheta[npn] = rhom * (p - (m + 1) * x * p1) / y * T(prefactor[npn]) * eim;

        rhom /= rho;
        T rhon = rhom;

        for (int n = m + 1; n != 2 * P; ++n)
        {
            int npm = n * n + n + m;
            int nmm = n * n + n - m;

            Ynm[npm] = rhon * p * T(prefactor[npm]) * eim;
            Ynm[nmm] = thrust::conj(Ynm[npm]);

            T p2 = p1;
            p1   = p;
            p    = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);

            YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) / y * T(prefactor[npm]) * eim;
            rhon /= rho;
        }

        pn = -pn * fact * y;
        fact += 2;
    }
}

// ---------------------------------------------------------------------------
// P2M — Particle to Multipole (device, with stride for warp-parallel reduction)
// ---------------------------------------------------------------------------

template<int stride = 1, class T1, class T2>
__device__ void P2MGpu(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
                       const Vec4<T1>& center, GpuSphericalMultipole<T1>& multipole, const double* prefactor)
{
    constexpr int PP = ExpansionOrder;

    GpuComplex<T1> Ynm_buf[4 * PP * PP];
    GpuComplex<T1> YnmTheta_buf[4 * PP * PP];

    for (auto& v : multipole)
        v = GpuComplex<T1>(0, 0);

    for (LocalIndex i = begin; i < end; i += stride)
    {
        Vec3<T1> dist{x[i] - center[0], y[i] - center[1], z[i] - center[2]};
        T1       rho, alpha, beta;
        cart2sphGpu(rho, alpha, beta, dist);
        evalMultipoleGpu<PP>(Ynm_buf, YnmTheta_buf, prefactor, rho, alpha, -beta);

        for (int n = 0; n < PP; ++n)
        {
            for (int k = 0; k <= n; ++k)
            {
                int nm  = n * n + n + k;
                int nms = n * (n + 1) / 2 + k;
                multipole[nms] += T1(m[i]) * Ynm_buf[nm];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M2M — Multipole to Multipole (device, single child contribution)
// ---------------------------------------------------------------------------

template<class T, class Tm>
__device__ void M2MGpu(const Vec4<T>& Xout, const Vec4<T>& Xchild, const GpuSphericalMultipole<Tm>& Mchild,
                       GpuSphericalMultipole<Tm>& Mout, const double* prefactor, const double* Anm)
{
    constexpr int       PP = ExpansionOrder;
    const GpuComplex<Tm> I(0, 1);

    GpuComplex<Tm> Ynm_buf[4 * PP * PP];
    GpuComplex<Tm> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{Xout[0] - Xchild[0], Xout[1] - Xchild[1], Xout[2] - Xchild[2]};
    T       rho, alpha, beta;
    cart2sphGpu(rho, alpha, beta, dist);
    evalMultipoleGpu<PP>(Ynm_buf, YnmTheta_buf, prefactor, Tm(rho), Tm(alpha), -Tm(beta));

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int            jk  = j * j + j + k;
            int            jks = j * (j + 1) / 2 + k;
            GpuComplex<Tm> M(0, 0);

            for (int n = 0; n <= j; ++n)
            {
                for (int m = -n; m <= min(k - 1, n); ++m)
                {
                    if (j - n >= k - m)
                    {
                        int jnkm  = (j - n) * (j - n) + j - n + k - m;
                        int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
                        int nm    = n * n + n + m;
                        int absm  = m < 0 ? -m : m;
                        M += Mchild[jnkms] * thrust::pow(I, Tm(m - absm)) * Ynm_buf[nm] *
                             Tm(oddevenGpu(n) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                    }
                }
                for (int m = k; m <= n; ++m)
                {
                    if (j - n >= m - k)
                    {
                        int jnkm  = (j - n) * (j - n) + j - n + k - m;
                        int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
                        int nm    = n * n + n + m;
                        M += thrust::conj(Mchild[jnkms]) * Ynm_buf[nm] *
                             Tm(oddevenGpu(k + n + m) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                    }
                }
            }

            Mout[jks] += M;
        }
    }
}

// ---------------------------------------------------------------------------
// M2L — Multipole to Local (device, with atomic accumulation)
// ---------------------------------------------------------------------------

template<class T>
__device__ void M2LGpu(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
                       const GpuSphericalMultipole<T>& multipole, GpuSphericalLocalExpansion<T>* local,
                       const double* prefactor, const double* Anm, const GpuComplex<double>* Cnm)
{
    constexpr int PP  = ExpansionOrder;
    constexpr int PP2 = PP * PP;

    GpuComplex<T> Ynm_buf[4 * PP * PP];
    GpuComplex<T> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{targetCenter[0] - sourceCenter[0], targetCenter[1] - sourceCenter[1],
                 targetCenter[2] - sourceCenter[2]};
    T       rho, alpha, beta;
    cart2sphGpu(rho, alpha, beta, dist);
    evalLocalGpu<PP>(Ynm_buf, YnmTheta_buf, prefactor, rho, alpha, beta);

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int           jk  = j * j + j + k;
            int           jks = j * (j + 1) / 2 + k;
            GpuComplex<T> L(0, 0);

            for (int n = 0; n < PP; ++n)
            {
                for (int m = -n; m < 0; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 - m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += thrust::conj(GpuComplex<T>(multipole[nms])) * GpuComplex<T>(Cnm[jknm]) * Ynm_buf[jnkm];
                }
                for (int m = 0; m <= n; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 + m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += GpuComplex<T>(multipole[nms]) * GpuComplex<T>(Cnm[jknm]) * Ynm_buf[jnkm];
                }
            }
            atomicAddComplex(&((*local)[jks]), L);
        }
    }
}

// ---------------------------------------------------------------------------
// L2L — Local to Local (device, single child)
// ---------------------------------------------------------------------------

template<class T, class Tm>
__device__ void L2LGpu(const Vec4<T>& Xparent, const Vec4<T>& Xchild,
                       const GpuSphericalLocalExpansion<Tm>& Lparent, GpuSphericalLocalExpansion<Tm>& Lchild,
                       const double* prefactor, const double* Anm)
{
    constexpr int          PP = ExpansionOrder;
    const GpuComplex<Tm>   I(0, 1);

    GpuComplex<Tm> Ynm_buf[4 * PP * PP];
    GpuComplex<Tm> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{Xchild[0] - Xparent[0], Xchild[1] - Xparent[1], Xchild[2] - Xparent[2]};
    T       rho, alpha, beta;
    cart2sphGpu(rho, alpha, beta, dist);
    evalMultipoleGpu<PP>(Ynm_buf, YnmTheta_buf, prefactor, Tm(rho), Tm(alpha), Tm(beta));

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int            jk  = j * j + j + k;
            int            jks = j * (j + 1) / 2 + k;
            GpuComplex<Tm> L(0, 0);

            for (int n = j; n < PP; ++n)
            {
                for (int m = j + k - n; m < 0; ++m)
                {
                    int absm_k = m - k;
                    if (absm_k < 0) absm_k = -absm_k;
                    if (n - j >= absm_k)
                    {
                        int jnkm = (n - j) * (n - j) + n - j + m - k;
                        int nm   = n * n + n - m;
                        int nms  = n * (n + 1) / 2 - m;
                        L += thrust::conj(Lparent[nms]) * Ynm_buf[jnkm] *
                             Tm(oddevenGpu(k) * Anm[jnkm] * Anm[jk] / Anm[nm]);
                    }
                }
                for (int m = 0; m <= n; ++m)
                {
                    int absm_k = m - k;
                    if (absm_k < 0) absm_k = -absm_k;
                    if (n - j >= absm_k)
                    {
                        int jnkm = (n - j) * (n - j) + n - j + m - k;
                        int nm   = n * n + n + m;
                        int nms  = n * (n + 1) / 2 + m;
                        int diff = m - k;
                        int absd = diff < 0 ? -diff : diff;
                        L += Lparent[nms] * thrust::pow(I, Tm(diff - absd)) * Ynm_buf[jnkm] *
                             Tm(Anm[jnkm] * Anm[jk] / Anm[nm]);
                    }
                }
            }
            Lchild[jks] += L;
        }
    }
}

// ---------------------------------------------------------------------------
// L2P — Local to Particle (device)
// ---------------------------------------------------------------------------

template<class Ta, class Tc, class Tm>
__device__ Vec4<Ta> L2PGpu(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
                           const GpuSphericalLocalExpansion<Tm>& local, const double* prefactor)
{
    constexpr int        PP = ExpansionOrder;
    const GpuComplex<Ta> I(0, 1);

    GpuComplex<Ta> Ynm_buf[4 * PP * PP];
    GpuComplex<Ta> YnmTheta_buf[4 * PP * PP];

    Vec3<Tc> dist{target[0] - center[0], target[1] - center[1], target[2] - center[2]};
    Ta       r, theta, phi;
    cart2sphGpu(r, theta, phi, dist);
    evalMultipoleGpu<PP>(Ynm_buf, YnmTheta_buf, prefactor, r, theta, phi);

    Ta       potential = 0;
    Vec3<Ta> spherical{0, 0, 0};

    for (int n = 0; n < PP; ++n)
    {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;

        potential += (GpuComplex<Ta>(local[nms]) * Ynm_buf[nm]).real();
        spherical[0] += (GpuComplex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n;
        spherical[1] += (GpuComplex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();

        for (int m = 1; m <= n; ++m)
        {
            nm  = n * n + n + m;
            nms = n * (n + 1) / 2 + m;

            potential += Ta(2) * (GpuComplex<Ta>(local[nms]) * Ynm_buf[nm]).real();
            spherical[0] += Ta(2) * (GpuComplex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n;
            spherical[1] += Ta(2) * (GpuComplex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();
            spherical[2] += Ta(2) * (GpuComplex<Ta>(local[nms]) * Ynm_buf[nm] * I).real() * m;
        }
    }

    Vec3<Ta> cartesian = sph2cartGpu(r, theta, phi, spherical);
    return acc + Vec4<Ta>{-potential, cartesian[0], cartesian[1], cartesian[2]};
}

// ---------------------------------------------------------------------------
// P2M kernel — compute leaf multipoles on GPU
// ---------------------------------------------------------------------------

template<int TPL, class T>
__global__ void computeLeafMultipolesGpuKernel(const T* x, const T* y, const T* z, const T* m,
                                                const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                                const LocalIndex* layout, const Vec4<T>* centers,
                                                GpuSphericalMultipole<T>* multipoles, GpuSphericalTables tables)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex leafIdx = tid / TPL;
    TreeNodeIndex internalIdx;

    GpuSphericalMultipole<T> mp_loc;
    for (auto& v : mp_loc)
        v = GpuComplex<T>(0, 0);

    if (leafIdx < numLeaves)
    {
        internalIdx = leafToInternal[leafIdx];
        auto com    = centers[internalIdx];
        P2MGpu<TPL>(x, y, z, m, layout[leafIdx] + threadIdx.x % TPL, layout[leafIdx + 1], com, mp_loc,
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
                               GpuSphericalMultipole<T>* d_multipoles, const GpuSphericalTables& tables)
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
                                            GpuSphericalMultipole<T>* multipoles, GpuSphericalTables tables)
{
    TreeNodeIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int     cellIdx = tid / 8 + firstCell;

    TreeNodeIndex firstChild = 0;
    if (cellIdx < lastCell) { firstChild = childOffsets[cellIdx]; }

    GpuSphericalMultipole<T> Mout;
    for (auto& v : Mout)
        v = GpuComplex<T>(0, 0);

    if (firstChild)
    {
        int child = firstChild + threadIdx.x % 8;
        M2MGpu(centers[cellIdx], centers[child], multipoles[child], Mout, tables.prefactor, tables.Anm);
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
                           const Vec4<T>* d_centers, GpuSphericalMultipole<T>* d_multipoles,
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
                          const Vec4<T>* centers, GpuSphericalLocalExpansion<T>* locals, GpuSphericalTables tables)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end) return;

    TreeNodeIndex firstChild = childOffsets[i];
    if (firstChild == 0) return; // leaf node

    for (int c = firstChild; c < firstChild + 8; ++c)
    {
        L2LGpu(centers[i], centers[c], locals[i], locals[c], tables.prefactor, tables.Anm);
    }
}

template<class T>
void downsweepLocalExpansionsGpu(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* d_childOffsets,
                                  const Vec4<T>* d_centers, GpuSphericalLocalExpansion<T>* d_locals,
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
                          const GpuSphericalLocalExpansion<T>* locals, const T* x, const T* y, const T* z,
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
        acc      = L2PGpu(acc, target, center, L, tables.prefactor);
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

template<int numWarps, class T>
__global__ void fmmDualTraversalKernel(const TreeNodeIndex* __restrict__ childOffsets,
                                       const Vec3<T>* __restrict__ geoCenters,
                                       const Vec3<T>* __restrict__ geoSizes,
                                       const Vec4<T>* __restrict__ centers,
                                       const GpuSphericalMultipole<T>* __restrict__ multipoles,
                                       GpuSphericalLocalExpansion<T>* __restrict__ locals,
                                       const TreeNodeIndex* __restrict__ internalToLeaf,
                                       const LocalIndex* __restrict__ layout,
                                       const T* __restrict__ x, const T* __restrict__ y,
                                       const T* __restrict__ z, const T* __restrict__ h, const T* __restrict__ m,
                                       T* __restrict__ ppot, T* __restrict__ pax, T* __restrict__ pay,
                                       T* __restrict__ paz, LocalIndex firstTarget,
                                       T invTheta, GpuSphericalTables tables,
                                       cstone::GlobalWorkQueue gq, cstone::GlobalTraversalQueue tq,
                                       unsigned* nProd,
                                       unsigned* d_m2lCount, unsigned* d_p2pCount)
{
    auto continuation = [geoCenters, geoSizes, invTheta] __device__(TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> centerA = geoCenters[a];
        Vec3<T> sizeA   = geoSizes[a];
        Vec3<T> centerB = geoCenters[b];
        Vec3<T> sizeB   = geoSizes[b];

        Vec3<T> d     = cstone::minDistance(centerA, sizeA, centerB, sizeB);
        T       dist2 = norm2(d);

        T lA        = T(2) * max(max(sizeA[0], sizeA[1]), sizeA[2]);
        T lB        = T(2) * max(max(sizeB[0], sizeB[1]), sizeB[2]);
        T threshold = max(lA, lB) * invTheta;

        return dist2 < threshold * threshold;
    };

    auto m2l = [centers, multipoles, locals, tables, d_m2lCount] __device__(TreeNodeIndex a, TreeNodeIndex b)
    {
        atomicAdd(d_m2lCount, 1u);
        M2LGpu(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], &locals[a], tables.prefactor,
               tables.Anm, tables.Cnm);
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
// Host orchestration
// ---------------------------------------------------------------------------

template<class T, class KeyType>
void computeGravityFMMGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                           const TreeNodeIndex* internalToLeaf,
                           std::span<const TreeNodeIndex> leafToInternalMap,
                           std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                           const SphericalMultipole<T>* multipoles, const LocalIndex* layout,
                           TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                           const T* z, const T* h, const T* m, const cstone::Box<T>& box, float theta, float G,
                           T* ugrav, T* ax, T* ay, T* az, T* ugravTot, LocalIndex numParticles)
{
    TreeNodeIndex numNodes     = levelRange.back();
    TreeNodeIndex numLeaves    = TreeNodeIndex(leafToInternalMap.size());
    T             invTheta     = T(1) / T(theta);
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
        h_geoSizes[i]     = sz;
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
    GpuSphericalMultipole<T>* d_multipoles;
    checkGpuErrors(cudaMalloc(&d_multipoles, numNodes * sizeof(GpuSphericalMultipole<T>)));
    checkGpuErrors(cudaMemset(d_multipoles, 0, numNodes * sizeof(GpuSphericalMultipole<T>)));

    computeLeafMultipolesGpu(d_x, d_y, d_z, d_m, d_leafToInternal, numLeaves, d_layout, d_centers, d_multipoles,
                             tables);

    upsweepMultipolesGpu(levelRange, d_childOffsets, d_centers, d_multipoles, tables);
    checkGpuErrors(cudaDeviceSynchronize());

    // 6. Allocate + zero-init device locals and particle accumulators
    GpuSphericalLocalExpansion<T>* d_locals;
    checkGpuErrors(cudaMalloc(&d_locals, numNodes * sizeof(GpuSphericalLocalExpansion<T>)));
    checkGpuErrors(cudaMemset(d_locals, 0, numNodes * sizeof(GpuSphericalLocalExpansion<T>)));

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

    // Query max co-resident blocks to avoid occupancy deadlock.
    // All launched blocks must be schedulable concurrently so every block
    // decrements numActiveProducers and the kernel can terminate.
    // Future (Blackwell sm_100+): Use Cluster Launch Control with try_cancel
    // to over-subscribe the grid and dynamically steal work from unscheduled
    // clusters, eliminating the need for conservative grid sizing.
    // See cudaLaunchAttributeClusterSchedulingPolicyPreference / cudaTryCancel.
    unsigned maxBlocks = cstone::maxConcurrentBlocks(
        fmmDualTraversalKernel<numWarps, T>, threadsPerBlock, kBlocksPerCluster);
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

    // 10. Allocate interaction counters for diagnostics
    unsigned* d_m2lCount;
    unsigned* d_p2pCount;
    checkGpuErrors(cudaMalloc(&d_m2lCount, sizeof(unsigned)));
    checkGpuErrors(cudaMalloc(&d_p2pCount, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_m2lCount, 0, sizeof(unsigned)));
    checkGpuErrors(cudaMemset(d_p2pCount, 0, sizeof(unsigned)));

    // 11. Launch dual traversal
    checkGpuErrors(cudaLaunchKernelEx(&dualCfg, fmmDualTraversalKernel<numWarps, T>, d_childOffsets, d_geoCenters,
                                      d_geoSizes, d_centers, d_multipoles, d_locals, d_internalToLeaf, d_layout, d_x,
                                      d_y, d_z, d_h, d_m, d_ppot, d_pax, d_pay, d_paz, firstTarget, invTheta, tables,
                                      gq, tq, d_nProd, d_m2lCount, d_p2pCount));
    checkGpuErrors(cudaDeviceSynchronize());

    // Print interaction counts for diagnostics
    unsigned h_m2l, h_p2p;
    checkGpuErrors(cudaMemcpy(&h_m2l, d_m2lCount, sizeof(unsigned), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(&h_p2p, d_p2pCount, sizeof(unsigned), cudaMemcpyDeviceToHost));
    printf("[FMM GPU] M2L calls: %u, P2P calls: %u\n", h_m2l, h_p2p);

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

    cudaFree(d_m2lCount);
    cudaFree(d_p2pCount);
}

} // namespace fmm
