/*
 * Spherical harmonic multipole kernels for Barnes-Hut / FMM
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Spherical harmonic multipole P2M, M2M, M2L, M2P, L2L, L2P kernels
 *
 * Ported from exafmm (Rio Yokota) Laplace kernel to the sphexa/ryoanji
 * multipole kernel interface. Uses compact storage with NTERM = P*(P+1)/2
 * terms (21 for P=6) instead of exafmm's strided 3*NCOEF layout.
 */

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif

#include "cstone/util/array.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/kernel.hpp"

#include "fmm_types.hpp"

namespace fmm
{

using ryoanji::Vec3;
using ryoanji::Vec4;
using ryoanji::LocalIndex;
using ryoanji::TreeNodeIndex;

// ---------------------------------------------------------------------------
// Constants and types
// ---------------------------------------------------------------------------

constexpr int ExpansionOrder = 6;

template<int P>
constexpr int Nterm = P * (P + 1) / 2;

//! @brief Complex type alias: thrust::complex on device, std::complex on host
template<class T>
#if defined(__CUDACC__) || defined(__HIPCC__)
using Complex = thrust::complex<T>;
#else
using Complex = std::complex<T>;
#endif

/*! @brief Spherical harmonic multipole type
 *
 * Defined as a struct (not a type alias) so that ADL finds P2M, M2M
 * in namespace fmm when called from templated code like computeLeafMultipoles.
 * Uses Complex<T> which maps to thrust::complex<T> in .cu and std::complex<T> in .cpp.
 * Both have identical memory layout.
 */
template<class T, int P = ExpansionOrder>
struct SphericalMultipole : util::array<Complex<T>, Nterm<P>>
{
    using Base = util::array<Complex<T>, Nterm<P>>;
    using Base::operator[];
};

/*! @brief Spherical harmonic local expansion type
 *
 * Same compact storage as SphericalMultipole. Separate struct for ADL
 * so that L2L, L2P are found in namespace fmm independently.
 */
template<class T, int P = ExpansionOrder>
struct SphericalLocalExpansion : util::array<Complex<T>, Nterm<P>>
{
    using Base = util::array<Complex<T>, Nterm<P>>;
    using Base::operator[];
};

HOST_DEVICE_FUN constexpr int oddeven(int n) { return (n & 1) ? -1 : 1; }

// ---------------------------------------------------------------------------
// Precomputed tables
// ---------------------------------------------------------------------------

template<int P>
struct SphericalTables
{
    static constexpr int P2 = P * P;
    static constexpr int P4 = P2 * P2;

    std::array<double, 4 * P2> prefactor;
    std::array<double, 4 * P2> Anm;
    std::array<std::complex<double>, P4> Cnm; //!< M2L translation matrix C_{jk,nm}

    SphericalTables()
    {
        prefactor.fill(0.0);
        Anm.fill(0.0);
        Cnm.fill(std::complex<double>(0.0, 0.0));

        for (int n = 0; n < 2 * P; ++n)
        {
            for (int m = -n; m <= n; ++m)
            {
                int    nm    = n * n + n + m;
                int    nabsm = std::abs(m);
                double fnmm  = 1.0;
                for (int i = 1; i <= n - m; ++i)
                    fnmm *= i;
                double fnpm = 1.0;
                for (int i = 1; i <= n + m; ++i)
                    fnpm *= i;
                double fnma = 1.0;
                for (int i = 1; i <= n - nabsm; ++i)
                    fnma *= i;
                double fnpa = 1.0;
                for (int i = 1; i <= n + nabsm; ++i)
                    fnpa *= i;

                prefactor[nm] = std::sqrt(fnma / fnpa);
                Anm[nm]       = oddeven(n) / std::sqrt(fnmm * fnpm);
            }
        }

        // Precompute M2L translation matrix (ported from exafmm kernel.h::preCalculation)
        const std::complex<double> I(0.0, 1.0);
        for (int j = 0, jk = 0, jknm = 0; j != P; ++j)
        {
            for (int k = -j; k <= j; ++k, ++jk)
            {
                for (int n = 0, nm = 0; n != P; ++n)
                {
                    for (int m = -n; m <= n; ++m, ++nm, ++jknm)
                    {
                        int jnkm    = (j + n) * (j + n) + j + n + m - k;
                        Cnm[jknm] = std::pow(I, double(std::abs(k - m) - std::abs(k) - std::abs(m)))
                                    * (double(oddeven(j)) * Anm[nm] * Anm[jk] / Anm[jnkm]);
                    }
                }
            }
        }
    }

    static const SphericalTables<P>& instance()
    {
        static const SphericalTables<P> tables;
        return tables;
    }
};

// ---------------------------------------------------------------------------
// Coordinate conversions
// ---------------------------------------------------------------------------

template<class T>
HOST_DEVICE_FUN void cart2sph(T& r, T& theta, T& phi, Vec3<T> dist)
{
    using std::sqrt; using std::acos; using std::abs; using std::atan;
    constexpr T EPS = T(1e-6);
    r               = sqrt(norm2(dist)) + EPS;
    theta           = acos(dist[2] / r);
    if (abs(dist[0]) + abs(dist[1]) < EPS) { phi = 0; }
    else if (abs(dist[0]) < EPS) { phi = dist[1] / abs(dist[1]) * T(M_PI) * T(0.5); }
    else if (dist[0] > 0) { phi = atan(dist[1] / dist[0]); }
    else { phi = atan(dist[1] / dist[0]) + T(M_PI); }
}

template<class T>
HOST_DEVICE_FUN Vec3<T> sph2cart(T r, T theta, T phi, Vec3<T> spherical)
{
    using std::sin; using std::cos;
    T st = sin(theta);
    T ct = cos(theta);
    T sp = sin(phi);
    T cp = cos(phi);

    return {st * cp * spherical[0] + ct * cp / r * spherical[1] - sp / r / st * spherical[2],
            st * sp * spherical[0] + ct * sp / r * spherical[1] + cp / r / st * spherical[2],
            ct * spherical[0] - st / r * spherical[1]};
}

// ---------------------------------------------------------------------------
// Solid harmonics evaluation
// ---------------------------------------------------------------------------

//! @brief Evaluate solid harmonics r^n Y_n^m (regular, for P2M/M2M)
template<int P, class T, class Cmplx>
HOST_DEVICE_FUN void evalMultipole(Cmplx* Ynm, Cmplx* YnmTheta, const double* prefactor, T rho, T alpha, T beta)
{
    using std::cos; using std::sin;
    const Cmplx I(0, 1);
    T           x    = cos(alpha);
    T           y    = sin(alpha);
    T           fact = 1;
    T           pn   = 1;
    T           rhom = 1;

    for (int m = 0; m != P; ++m)
    {
        Cmplx eim = exp(I * T(m * beta));
        T     p   = pn;
        int   npn = m * m + 2 * m;
        int   nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = conj(Ynm[npn]);

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
            Ynm[nmm] = conj(Ynm[npm]);

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

//! @brief Evaluate singular harmonics r^{-n-1} Y_n^m (for M2L/M2P)
template<int P, class T, class Cmplx>
HOST_DEVICE_FUN void evalLocal(Cmplx* Ynm, Cmplx* YnmTheta, const double* prefactor, T rho, T alpha, T beta)
{
    using std::cos; using std::sin;
    const Cmplx I(0, 1);
    T           x    = cos(alpha);
    T           y    = sin(alpha);
    T           fact = 1;
    T           pn   = 1;
    T           rhom = T(1) / rho;

    for (int m = 0; m != 2 * P; ++m)
    {
        Cmplx eim = exp(I * T(m * beta));
        T     p   = pn;
        int   npn = m * m + 2 * m;
        int   nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = conj(Ynm[npn]);

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
            Ynm[nmm] = conj(Ynm[npm]);

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
// P2M — Particle to Multipole
// ---------------------------------------------------------------------------

//! @brief Core P2M with explicit table pointer — works on host and device
template<int stride = 1, class T1, class T2>
HOST_DEVICE_FUN void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
         const Vec4<T1>& center, SphericalMultipole<T1>& multipole, const double* prefactor)
{
    constexpr int PP = ExpansionOrder;

    Complex<T1> Ynm_buf[4 * PP * PP];
    Complex<T1> YnmTheta_buf[4 * PP * PP];

    for (auto& v : multipole)
        v = Complex<T1>(0, 0);

    for (LocalIndex i = begin; i < end; i += stride)
    {
        Vec3<T1> dist{x[i] - center[0], y[i] - center[1], z[i] - center[2]};
        T1       rho, alpha, beta;
        cart2sph(rho, alpha, beta, dist);
        evalMultipole<PP>(Ynm_buf, YnmTheta_buf, prefactor, rho, alpha, -beta);

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

//! @brief CPU wrapper — gets tables from singleton
template<int stride = 1, class T1, class T2>
void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
         const Vec4<T1>& center, SphericalMultipole<T1>& multipole)
{
    P2M<stride>(x, y, z, m, begin, end, center, multipole, SphericalTables<ExpansionOrder>::instance().prefactor.data());
}

// ---------------------------------------------------------------------------
// M2M — Multipole to Multipole
// ---------------------------------------------------------------------------

//! @brief Core M2M with explicit table pointers — single child contribution, works on host and device
template<class T, class Tm>
HOST_DEVICE_FUN void M2M(const Vec4<T>& Xout, const Vec4<T>& Xchild, const SphericalMultipole<Tm>& Mchild,
         SphericalMultipole<Tm>& Mout, const double* prefactor, const double* Anm)
{
    constexpr int     PP = ExpansionOrder;
    const Complex<Tm> I(0, 1);

    Complex<Tm> Ynm_buf[4 * PP * PP];
    Complex<Tm> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{Xout[0] - Xchild[0], Xout[1] - Xchild[1], Xout[2] - Xchild[2]};
    T       rho, alpha, beta;
    cart2sph(rho, alpha, beta, dist);
    evalMultipole<PP>(Ynm_buf, YnmTheta_buf, prefactor, Tm(rho), Tm(alpha), -Tm(beta));

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int        jk  = j * j + j + k;
            int        jks = j * (j + 1) / 2 + k;
            Complex<Tm> M(0, 0);

            for (int n = 0; n <= j; ++n)
            {
                for (int m = -n; m <= ((k - 1 < n) ? k - 1 : n); ++m)
                {
                    if (j - n >= k - m)
                    {
                        int jnkm  = (j - n) * (j - n) + j - n + k - m;
                        int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
                        int nm    = n * n + n + m;
                        int absm  = m < 0 ? -m : m;
                        M += Mchild[jnkms] * pow(I, Tm(m - absm)) * Ynm_buf[nm] *
                             Tm(oddeven(n) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                    }
                }
                for (int m = k; m <= n; ++m)
                {
                    if (j - n >= m - k)
                    {
                        int jnkm  = (j - n) * (j - n) + j - n + k - m;
                        int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
                        int nm    = n * n + n + m;
                        M += conj(Mchild[jnkms]) * Ynm_buf[nm] *
                             Tm(oddeven(k + n + m) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                    }
                }
            }

            Mout[jks] += M;
        }
    }
}

//! @brief CPU M2M wrapper — multi-child, gets tables from singleton
template<class T, class Tm>
void M2M(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const SphericalMultipole<Tm>* Msrc,
         SphericalMultipole<Tm>& Mout)
{
    const auto& tab = SphericalTables<ExpansionOrder>::instance();

    for (auto& v : Mout)
        v = Complex<Tm>(0, 0);

    for (int c = begin; c < end; ++c)
    {
        M2M(Xout, Xsrc[c], Msrc[c], Mout, tab.prefactor.data(), tab.Anm.data());
    }
}

// ---------------------------------------------------------------------------
// M2L — Multipole to Local
// ---------------------------------------------------------------------------

//! @brief Core M2L with explicit table pointers — works on host and device
template<class T>
HOST_DEVICE_FUN void M2L(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
         const SphericalMultipole<T>& multipole, SphericalLocalExpansion<T>& local,
         const double* prefactor, const double* /*Anm*/, const Complex<double>* Cnm)
{
    constexpr int PP  = ExpansionOrder;
    constexpr int PP2 = PP * PP;

    Complex<T> Ynm_buf[4 * PP * PP];
    Complex<T> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{targetCenter[0] - sourceCenter[0], targetCenter[1] - sourceCenter[1],
                 targetCenter[2] - sourceCenter[2]};
    T       rho, alpha, beta;
    cart2sph(rho, alpha, beta, dist);
    evalLocal<PP>(Ynm_buf, YnmTheta_buf, prefactor, rho, alpha, beta);

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int        jk  = j * j + j + k;
            int        jks = j * (j + 1) / 2 + k;
            Complex<T> L(0, 0);

            for (int n = 0; n < PP; ++n)
            {
                for (int m = -n; m < 0; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 - m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += conj(Complex<T>(multipole[nms])) * Complex<T>(Cnm[jknm]) * Ynm_buf[jnkm];
                }
                for (int m = 0; m <= n; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 + m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += Complex<T>(multipole[nms]) * Complex<T>(Cnm[jknm]) * Ynm_buf[jnkm];
                }
            }
            local[jks] += L;
        }
    }
}

//! @brief CPU wrapper — gets tables from singleton
template<class T>
void M2L(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
         const SphericalMultipole<T>& multipole, SphericalLocalExpansion<T>& local)
{
    const auto& tab = SphericalTables<ExpansionOrder>::instance();
    M2L(targetCenter, sourceCenter, multipole, local, tab.prefactor.data(), tab.Anm.data(),
        reinterpret_cast<const Complex<double>*>(tab.Cnm.data()));
}

// ---------------------------------------------------------------------------
// L2L — Local to Local
// ---------------------------------------------------------------------------

//! @brief Core L2L with explicit table pointers — single child, works on host and device
template<class T, class Tm>
HOST_DEVICE_FUN void L2L(const Vec4<T>& Xparent, const Vec4<T>& Xchild,
         const SphericalLocalExpansion<Tm>& Lparent, SphericalLocalExpansion<Tm>& Lchild,
         const double* prefactor, const double* Anm)
{
    constexpr int     PP = ExpansionOrder;
    const Complex<Tm> I(0, 1);

    Complex<Tm> Ynm_buf[4 * PP * PP];
    Complex<Tm> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{Xchild[0] - Xparent[0], Xchild[1] - Xparent[1], Xchild[2] - Xparent[2]};
    T       rho, alpha, beta;
    cart2sph(rho, alpha, beta, dist);
    evalMultipole<PP>(Ynm_buf, YnmTheta_buf, prefactor, Tm(rho), Tm(alpha), Tm(beta));

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int         jk  = j * j + j + k;
            int         jks = j * (j + 1) / 2 + k;
            Complex<Tm> L(0, 0);

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
                        L += conj(Lparent[nms]) * Ynm_buf[jnkm] *
                             Tm(oddeven(k) * Anm[jnkm] * Anm[jk] / Anm[nm]);
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
                        L += Lparent[nms] * pow(I, Tm(diff - absd)) * Ynm_buf[jnkm] *
                             Tm(Anm[jnkm] * Anm[jk] / Anm[nm]);
                    }
                }
            }
            Lchild[jks] += L;
        }
    }
}

//! @brief CPU L2L wrapper — multi-child, gets tables from singleton
template<class T, class Tm>
void L2L(int childBegin, int childEnd, const Vec4<T>& Xparent, const Vec4<T>* Xchildren,
         const SphericalLocalExpansion<Tm>& Lparent, SphericalLocalExpansion<Tm>* Lchildren)
{
    const auto& tab = SphericalTables<ExpansionOrder>::instance();

    for (int c = childBegin; c < childEnd; ++c)
    {
        L2L(Xparent, Xchildren[c], Lparent, Lchildren[c], tab.prefactor.data(), tab.Anm.data());
    }
}

// ---------------------------------------------------------------------------
// L2P — Local to Particle
// ---------------------------------------------------------------------------

//! @brief Core L2P with explicit table pointer — works on host and device
template<class Ta, class Tc, class Tm>
HOST_DEVICE_FUN Vec4<Ta> L2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
         const SphericalLocalExpansion<Tm>& local, const double* prefactor)
{
    constexpr int     PP = ExpansionOrder;
    const Complex<Ta> I(0, 1);

    Complex<Ta> Ynm_buf[4 * PP * PP];
    Complex<Ta> YnmTheta_buf[4 * PP * PP];

    Vec3<Tc> dist{target[0] - center[0], target[1] - center[1], target[2] - center[2]};
    Ta       r, theta, phi;
    cart2sph(r, theta, phi, dist);
    evalMultipole<PP>(Ynm_buf, YnmTheta_buf, prefactor, r, theta, phi);

    Ta       potential = 0;
    Vec3<Ta> spherical{0, 0, 0};

    for (int n = 0; n < PP; ++n)
    {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;

        potential += (Complex<Ta>(local[nms]) * Ynm_buf[nm]).real();
        spherical[0] += (Complex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n;
        spherical[1] += (Complex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();

        for (int m = 1; m <= n; ++m)
        {
            nm  = n * n + n + m;
            nms = n * (n + 1) / 2 + m;

            potential += Ta(2) * (Complex<Ta>(local[nms]) * Ynm_buf[nm]).real();
            spherical[0] += Ta(2) * (Complex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n;
            spherical[1] += Ta(2) * (Complex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();
            spherical[2] += Ta(2) * (Complex<Ta>(local[nms]) * Ynm_buf[nm] * I).real() * m;
        }
    }

    Vec3<Ta> cartesian = sph2cart(r, theta, phi, spherical);

    return acc + Vec4<Ta>{-potential, cartesian[0], cartesian[1], cartesian[2]};
}

//! @brief CPU wrapper — gets tables from singleton
template<class Ta, class Tc, class Tm>
Vec4<Ta> L2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center, const SphericalLocalExpansion<Tm>& local)
{
    return L2P(acc, target, center, local, SphericalTables<ExpansionOrder>::instance().prefactor.data());
}

// ---------------------------------------------------------------------------
// L2L downsweep — propagate local expansions from root to leaves
// ---------------------------------------------------------------------------

template<class T>
void downsweepLocalExpansions(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* childOffsets,
                              const cstone::SourceCenterType<T>* centers, SphericalLocalExpansion<T>* locals)
{
    int numLevels = int(levelRange.size()) - 1;

    for (int currentLevel = 0; currentLevel < numLevels; ++currentLevel)
    {
        TreeNodeIndex start = levelRange[currentLevel];
        TreeNodeIndex end   = levelRange[currentLevel + 1];

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = start; i < end; ++i)
        {
            TreeNodeIndex firstChild = childOffsets[i];
            if (firstChild != 0) // internal node
            {
                L2L(firstChild, firstChild + 8, centers[i], centers, locals[i], locals);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full FMM gravity computation via dual traversal
// ---------------------------------------------------------------------------

template<MacVariant macType = ScalarMac, class T, class KeyType, class Th, class Tm>
void computeGravityFMM(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                        const TreeNodeIndex* internalToLeaf,
                        std::span<const TreeNodeIndex> leafToInternalMap,
                        std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                        const SphericalMultipole<T>* multipoles, const LocalIndex* layout,
                        TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                        const T* z, const Th* h, const Tm* m, const cstone::Box<T>& box, float G,
                        float invTheta, Th* ugrav, Th* ax, Th* ay, Th* az, T* ugravTot)
{
    TreeNodeIndex numNodes = levelRange.back();

    // 0. Compute geometric node centers and sizes for the dual-traversal MAC
    std::vector<Vec3<T>> geoCenters(numNodes);
    std::vector<Vec3<T>> geoSizes(numNodes);
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        KeyType  prefix   = prefixes[i];
        KeyType  startKey = cstone::decodePlaceholderBit(prefix);
        unsigned level    = cstone::decodePrefixLength(prefix) / 3;
        auto     nodeBox  = cstone::sfcIBox(cstone::sfcKey(startKey), level);
        auto [center, sz] = cstone::centerAndSize<KeyType>(nodeBox, box);
        geoCenters[i]     = center;
        geoSizes[i]       = sz * T(invTheta);
    }

    // 1. Allocate local expansions (zero-initialized)
    std::vector<SphericalLocalExpansion<T>> locals(numNodes);
    for (auto& L : locals)
        for (auto& v : L)
            v = std::complex<T>(0, 0);

    // 3. Allocate per-particle P2P + L2P accumulators
    LocalIndex firstTarget = layout[firstLeafIndex];
    LocalIndex lastTarget  = layout[lastLeafIndex];
    LocalIndex numTargets  = lastTarget - firstTarget;

    std::vector<T> pax(numTargets, 0), pay(numTargets, 0), paz(numTargets, 0), ppot(numTargets, 0);

    // 4. Dual traversal: M2L for well-separated pairs, P2P for nearby leaf-leaf pairs
    auto continuation = [centers, &geoCenters, &geoSizes](TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        if constexpr (macType == DirectionalMac)
        {
            return cstone::evaluateMac(util::makeVec3(centers[a]), centers[a][3],
                                        geoCenters[b], geoSizes[b]) ||
                   cstone::evaluateMac(util::makeVec3(centers[b]), centers[b][3],
                                        geoCenters[a], geoSizes[a]);
        }
        else
        {
            return cstone::evaluateMac(util::makeVec3(centers[a]), centers[a][3],
                                        geoCenters[b], geoSizes[b]);
        }
    };

    auto m2lCallback = [&centers, &multipoles, &locals](TreeNodeIndex a, TreeNodeIndex b)
    {
        M2L(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], locals[a]);
    };

    auto p2pCallback = [internalToLeaf, layout, x, y, z, h, m, &pax, &pay, &paz, &ppot,
                        firstTarget](TreeNodeIndex a, TreeNodeIndex b)
    {
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
            ppot[ti] += acc[0];
            pax[ti] += acc[1];
            pay[ti] += acc[2];
            paz[ti] += acc[3];
        }
    };

    TreeNodeIndex rootFirstChild = childOffsets[0];
    if (rootFirstChild == 0)
    {
        // Root is a leaf — fall back to serial
        cstone::dualTraversal(childOffsets, TreeNodeIndex(0), TreeNodeIndex(0), continuation, m2lCallback, p2pCallback);
    }
    else
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 8; ++i)
        {
            cstone::dualTraversal(childOffsets, rootFirstChild + i, TreeNodeIndex(0), continuation, m2lCallback,
                                  p2pCallback);
        }
    }

    // 5. L2L downsweep
    downsweepLocalExpansions<T>(levelRange, childOffsets, centers, locals.data());

    // 6. L2P: apply local expansion at each leaf to its particles
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex leafIdx = firstLeafIndex; leafIdx < lastLeafIndex; ++leafIdx)
    {
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
            acc      = L2P(acc, target, center, L);
            ppot[ti] = acc[0];
            pax[ti]  = acc[1];
            pay[ti]  = acc[2];
            paz[ti]  = acc[3];
        }
    }

    // 7. Apply G and accumulate into output arrays
    T ugravLoc = 0;
    for (LocalIndex t = firstTarget; t < lastTarget; ++t)
    {
        LocalIndex ti = t - firstTarget;
        auto       u  = G * m[t] * ppot[ti];
        ugravLoc += u;
        if (ugrav) { ugrav[t] += u; }
        ax[t] += G * pax[ti];
        ay[t] += G * pay[ti];
        az[t] += G * paz[ti];
    }

    *ugravTot += T(0.5) * ugravLoc;
}

} // namespace fmm
