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
#include <atomic>
#include <cmath>
#include <complex>
#include <vector>

#include "cstone/util/array.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/macs.hpp"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/kernel.hpp"

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

/*! @brief Spherical harmonic multipole type
 *
 * Defined as a struct (not a type alias) so that ADL finds P2M, M2M, M2P
 * in namespace fmm when called from templated code like computeLeafMultipoles.
 */
template<class T, int P = ExpansionOrder>
struct SphericalMultipole : util::array<std::complex<T>, Nterm<P>>
{
    using Base = util::array<std::complex<T>, Nterm<P>>;
    using Base::operator[];
};

/*! @brief Spherical harmonic local expansion type
 *
 * Same compact storage as SphericalMultipole. Separate struct for ADL
 * so that L2L, L2P are found in namespace fmm independently.
 */
template<class T, int P = ExpansionOrder>
struct SphericalLocalExpansion : util::array<std::complex<T>, Nterm<P>>
{
    using Base = util::array<std::complex<T>, Nterm<P>>;
    using Base::operator[];
};

constexpr int oddeven(int n) { return (n & 1) ? -1 : 1; }

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
void cart2sph(T& r, T& theta, T& phi, Vec3<T> dist)
{
    constexpr T EPS = T(1e-6);
    r               = std::sqrt(norm2(dist)) + EPS;
    theta           = std::acos(dist[2] / r);
    if (std::abs(dist[0]) + std::abs(dist[1]) < EPS) { phi = 0; }
    else if (std::abs(dist[0]) < EPS) { phi = dist[1] / std::abs(dist[1]) * T(M_PI) * T(0.5); }
    else if (dist[0] > 0) { phi = std::atan(dist[1] / dist[0]); }
    else { phi = std::atan(dist[1] / dist[0]) + T(M_PI); }
}

template<class T>
Vec3<T> sph2cart(T r, T theta, T phi, Vec3<T> spherical)
{
    T st = std::sin(theta);
    T ct = std::cos(theta);
    T sp = std::sin(phi);
    T cp = std::cos(phi);

    return {st * cp * spherical[0] + ct * cp / r * spherical[1] - sp / r / st * spherical[2],
            st * sp * spherical[0] + ct * sp / r * spherical[1] + cp / r / st * spherical[2],
            ct * spherical[0] - st / r * spherical[1]};
}

// ---------------------------------------------------------------------------
// Solid harmonics evaluation
// ---------------------------------------------------------------------------

//! @brief Evaluate solid harmonics r^n Y_n^m (regular, for P2M/M2M)
template<int P, class T>
void evalMultipole(std::complex<T>* Ynm, std::complex<T>* YnmTheta, const double* prefactor, T rho, T alpha, T beta)
{
    const std::complex<T> I(0, 1);
    T                     x    = std::cos(alpha);
    T                     y    = std::sin(alpha);
    T                     fact = 1;
    T                     pn   = 1;
    T                     rhom = 1;

    for (int m = 0; m != P; ++m)
    {
        std::complex<T> eim = std::exp(I * T(m * beta));
        T               p   = pn;
        int             npn = m * m + 2 * m;
        int             nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = std::conj(Ynm[npn]);

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
            Ynm[nmm] = std::conj(Ynm[npm]);

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

//! @brief Evaluate singular harmonics r^{-n-1} Y_n^m (for M2P)
template<int P, class T>
void evalLocal(std::complex<T>* Ynm, std::complex<T>* YnmTheta, const double* prefactor, T rho, T alpha, T beta)
{
    const std::complex<T> I(0, 1);
    T                     x    = std::cos(alpha);
    T                     y    = std::sin(alpha);
    T                     fact = 1;
    T                     pn   = 1;
    T                     rhom = T(1) / rho;

    for (int m = 0; m != 2 * P; ++m)
    {
        std::complex<T> eim = std::exp(I * T(m * beta));
        T               p   = pn;
        int             npn = m * m + 2 * m;
        int             nmn = m * m;

        Ynm[npn] = rhom * p * T(prefactor[npn]) * eim;
        Ynm[nmn] = std::conj(Ynm[npn]);

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
            Ynm[nmm] = std::conj(Ynm[npm]);

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

template<int stride = 1, class T1, class T2>
void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
         const Vec4<T1>& center, SphericalMultipole<T1>& multipole)
{
    constexpr int PP  = ExpansionOrder;
    const auto&   tab = SphericalTables<PP>::instance();

    std::complex<T1> Ynm_buf[4 * PP * PP];
    std::complex<T1> YnmTheta_buf[4 * PP * PP];

    for (auto& v : multipole)
        v = std::complex<T1>(0, 0);

    for (LocalIndex i = begin; i < end; i += stride)
    {
        Vec3<T1> dist{x[i] - center[0], y[i] - center[1], z[i] - center[2]};
        T1       rho, alpha, beta;
        cart2sph(rho, alpha, beta, dist);
        evalMultipole<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), rho, alpha, -beta);

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
// M2M — Multipole to Multipole
// ---------------------------------------------------------------------------

template<class T, class Tm>
void M2M(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const SphericalMultipole<Tm>* Msrc,
         SphericalMultipole<Tm>& Mout)
{
    constexpr int           PP  = ExpansionOrder;
    const auto&             tab = SphericalTables<PP>::instance();
    const std::complex<Tm>  I(0, 1);

    std::complex<Tm> Ynm_buf[4 * PP * PP];
    std::complex<Tm> YnmTheta_buf[4 * PP * PP];

    for (auto& v : Mout)
        v = std::complex<Tm>(0, 0);

    for (int c = begin; c < end; ++c)
    {
        Vec3<T> dist{Xout[0] - Xsrc[c][0], Xout[1] - Xsrc[c][1], Xout[2] - Xsrc[c][2]};
        T       rho, alpha, beta;
        cart2sph(rho, alpha, beta, dist);
        evalMultipole<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), Tm(rho), Tm(alpha), -Tm(beta));

        for (int j = 0; j < PP; ++j)
        {
            for (int k = 0; k <= j; ++k)
            {
                int             jk  = j * j + j + k;
                int             jks = j * (j + 1) / 2 + k;
                std::complex<Tm> M(0, 0);

                for (int n = 0; n <= j; ++n)
                {
                    // m < k: use M[jnkms] directly
                    for (int m = -n; m <= std::min(k - 1, n); ++m)
                    {
                        if (j - n >= k - m)
                        {
                            int jnkm  = (j - n) * (j - n) + j - n + k - m;
                            int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
                            int nm    = n * n + n + m;
                            M += Msrc[c][jnkms] * std::pow(I, Tm(m - std::abs(m))) * Ynm_buf[nm] *
                                 Tm(oddeven(n) * tab.Anm[nm] * tab.Anm[jnkm] / tab.Anm[jk]);
                        }
                    }
                    // m >= k: use conj(M[jnkms])
                    for (int m = k; m <= n; ++m)
                    {
                        if (j - n >= m - k)
                        {
                            int jnkm  = (j - n) * (j - n) + j - n + k - m;
                            int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
                            int nm    = n * n + n + m;
                            M += std::conj(Msrc[c][jnkms]) * Ynm_buf[nm] *
                                 Tm(oddeven(k + n + m) * tab.Anm[nm] * tab.Anm[jnkm] / tab.Anm[jk]);
                        }
                    }
                }

                Mout[jks] += M;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// M2P — Multipole to Particle
// ---------------------------------------------------------------------------

template<class Ta, class Tc, class Tm>
Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center, const SphericalMultipole<Tm>& multipole)
{
    constexpr int          PP  = ExpansionOrder;
    const auto&            tab = SphericalTables<PP>::instance();
    const std::complex<Ta> I(0, 1);

    std::complex<Ta> Ynm_buf[4 * PP * PP];
    std::complex<Ta> YnmTheta_buf[4 * PP * PP];

    Vec3<Tc> dist{target[0] - center[0], target[1] - center[1], target[2] - center[2]};
    Ta       r, theta, phi;
    cart2sph(r, theta, phi, dist);
    evalLocal<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), r, theta, phi);

    Ta       potential = 0;
    Vec3<Ta> spherical{0, 0, 0};

    for (int n = 0; n < PP; ++n)
    {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;

        potential += (std::complex<Ta>(multipole[nms]) * Ynm_buf[nm]).real();
        spherical[0] -= (std::complex<Ta>(multipole[nms]) * Ynm_buf[nm]).real() / r * (n + 1);
        spherical[1] += (std::complex<Ta>(multipole[nms]) * YnmTheta_buf[nm]).real();

        for (int m = 1; m <= n; ++m)
        {
            nm  = n * n + n + m;
            nms = n * (n + 1) / 2 + m;

            potential += Ta(2) * (std::complex<Ta>(multipole[nms]) * Ynm_buf[nm]).real();
            spherical[0] -= Ta(2) * (std::complex<Ta>(multipole[nms]) * Ynm_buf[nm]).real() / r * (n + 1);
            spherical[1] += Ta(2) * (std::complex<Ta>(multipole[nms]) * YnmTheta_buf[nm]).real();
            spherical[2] += Ta(2) * (std::complex<Ta>(multipole[nms]) * Ynm_buf[nm] * I).real() * m;
        }
    }

    Vec3<Ta> cartesian = sph2cart(r, theta, phi, spherical);

    // exafmm Laplace gives potential = +M/r (electrostatic convention)
    // ryoanji gravity uses potential = -M/r, so negate the potential
    // Force/acceleration signs match (both attractive) — no change needed
    return acc + Vec4<Ta>{-potential, cartesian[0], cartesian[1], cartesian[2]};
}

// ---------------------------------------------------------------------------
// Barnes-Hut traversal for SphericalMultipole
// (Custom version that avoids the mp[Cqi::mass] > 0 check which
//  doesn't compile for std::complex types)
// ---------------------------------------------------------------------------

template<class T1, size_t N>
auto computeCenterAndSize(const util::array<Vec4<T1>, N>& target)
{
    Vec3<T1> tMin = util::makeVec3(target[0]);
    Vec3<T1> tMax = tMin;
    for (LocalIndex i = 1; i < N; ++i)
    {
        Vec3<T1> tp = util::makeVec3(target[i]);
        tMin        = min(tp, tMin);
        tMax        = max(tp, tMax);
    }

    Vec3<T1> center = (tMax + tMin) * T1(0.5);
    Vec3<T1> size   = (tMax - tMin) * T1(0.5);

    return std::make_tuple(center, size);
}

template<class T1, class Th, class Tm, size_t N>
void computeGravityGroup(const util::array<Vec4<T1>, N>& target, const TreeNodeIndex* childOffsets,
                         const TreeNodeIndex* parents, const TreeNodeIndex* internalToLeaf,
                         const cstone::SourceCenterType<T1>* centers, const SphericalMultipole<T1>* multipoles,
                         const LocalIndex* layout, const T1* x, const T1* y, const T1* z, const Th* h, const Tm* m,
                         Vec4<T1>* acc)
{
    Vec3<T1> targetCenter, targetSize;
    std::tie(targetCenter, targetSize) = computeCenterAndSize(target);

    auto descendOrM2P =
        [internalToLeaf, layout, centers, multipoles, &target, &targetCenter, &targetSize, acc](TreeNodeIndex idx)
    {
        const auto& com = centers[idx];
        const auto& mp  = multipoles[idx];

        bool violatesMac = cstone::evaluateMac(util::makeVec3(com), com[3], targetCenter, targetSize);

        if (!violatesMac)
        {
            for (LocalIndex k = 0; k < N; ++k)
            {
                acc[k] = M2P(acc[k], util::makeVec3(target[k]), util::makeVec3(com), mp);
            }
        }

        return violatesMac;
    };

    auto leafP2P = [internalToLeaf, layout, &target, x, y, z, h, m, acc](TreeNodeIndex idx)
    {
        TreeNodeIndex lidx        = internalToLeaf[idx];
        LocalIndex    firstSource = layout[lidx];
        LocalIndex    lastSource  = layout[lidx + 1];

        for (LocalIndex k = 0; k < N; ++k)
        {
            for (LocalIndex s = firstSource; s < lastSource; ++s)
            {
                acc[k] =
                    ryoanji::P2P(acc[k], util::makeVec3(target[k]), Vec3<T1>{x[s], y[s], z[s]}, m[s], Th(target[k][3]), h[s]);
            }
        }
    };

    cstone::singleTraversal(childOffsets, parents, descendOrM2P, leafP2P);
}

template<class T1, class T2, class Tm>
void computeGravity(const TreeNodeIndex* childOffsets, const TreeNodeIndex* parents,
                    const TreeNodeIndex* internalToLeaf, const cstone::SourceCenterType<T1>* macSpheres,
                    const SphericalMultipole<T1>* multipoles, const LocalIndex* layout, TreeNodeIndex firstLeafIndex,
                    TreeNodeIndex lastLeafIndex, const T1* x, const T1* y, const T1* z, const T2* h, const Tm* m,
                    const cstone::Box<T1>& box, float G, T2* ugrav, T2* ax, T2* ay, T2* az, T1* ugravTot,
                    int numShells = 0)
{
    constexpr LocalIndex groupSize   = 16;
    LocalIndex           firstTarget = layout[firstLeafIndex];
    LocalIndex           lastTarget  = layout[lastLeafIndex];

    T1 ugravLoc = 0.0;

#pragma omp parallel for reduction(+ : ugravLoc)
    for (LocalIndex i = firstTarget; i < lastTarget; i += groupSize)
    {
        util::array<Vec4<T1>, groupSize> targets, potAndAcc;

        LocalIndex groupSizeValid = std::min(groupSize, lastTarget - i);
        for (LocalIndex k = 0; k < groupSizeValid; ++k)
        {
            targets[k]   = {x[i + k], y[i + k], z[i + k], T1(h[i + k])};
            potAndAcc[k] = {0, 0, 0, 0};
        }

        for (int iz = -numShells; iz <= numShells; ++iz)
        {
            for (int iy = -numShells; iy <= numShells; ++iy)
            {
                for (int ix = -numShells; ix <= numShells; ++ix)
                {
                    Vec4<T1> pbcShift{ix * box.lx(), iy * box.ly(), iz * box.lz(), 0};

                    auto targetsShifted = targets;
                    for (auto& t_ : targetsShifted)
                    {
                        t_ -= pbcShift;
                    }

                    computeGravityGroup(targetsShifted, childOffsets, parents, internalToLeaf, macSpheres, multipoles,
                                        layout, x, y, z, h, m, potAndAcc.data());
                }
            }
        }

        for (LocalIndex k = 0; k < groupSizeValid; ++k)
        {
            auto u = G * m[i + k] * potAndAcc[k][0];
            ugravLoc += u;
            if (ugrav) { ugrav[i + k] += u; }
            ax[i + k] += G * potAndAcc[k][1];
            ay[i + k] += G * potAndAcc[k][2];
            az[i + k] += G * potAndAcc[k][3];
        }
    }

    *ugravTot += T1(0.5) * ugravLoc;
}

// ---------------------------------------------------------------------------
// M2L — Multipole to Local
// ---------------------------------------------------------------------------

//! @brief Translate a source multipole expansion into a target local expansion
template<class T>
void M2L(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
         const SphericalMultipole<T>& multipole, SphericalLocalExpansion<T>& local)
{
    constexpr int PP  = ExpansionOrder;
    constexpr int PP2 = PP * PP;
    const auto&   tab = SphericalTables<PP>::instance();

    std::complex<T> Ynm_buf[4 * PP * PP];
    std::complex<T> YnmTheta_buf[4 * PP * PP];

    Vec3<T> dist{targetCenter[0] - sourceCenter[0], targetCenter[1] - sourceCenter[1],
                 targetCenter[2] - sourceCenter[2]};
    T       rho, alpha, beta;
    cart2sph(rho, alpha, beta, dist);
    evalLocal<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), rho, alpha, beta); // positive beta

    for (int j = 0; j < PP; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int             jk  = j * j + j + k;
            int             jks = j * (j + 1) / 2 + k;
            std::complex<T> L(0, 0);

            for (int n = 0; n < PP; ++n)
            {
                // m < 0: use conj(M[nms])
                for (int m = -n; m < 0; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 - m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += std::conj(std::complex<T>(multipole[nms])) * std::complex<T>(tab.Cnm[jknm]) * Ynm_buf[jnkm];
                }
                // m >= 0: use M[nms] directly
                for (int m = 0; m <= n; ++m)
                {
                    int nm   = n * n + n + m;
                    int nms  = n * (n + 1) / 2 + m;
                    int jknm = jk * PP2 + nm;
                    int jnkm = (j + n) * (j + n) + j + n + m - k;
                    L += std::complex<T>(multipole[nms]) * std::complex<T>(tab.Cnm[jknm]) * Ynm_buf[jnkm];
                }
            }
            local[jks] += L;
        }
    }
}

// ---------------------------------------------------------------------------
// L2L — Local to Local
// ---------------------------------------------------------------------------

//! @brief Translate a parent local expansion to its children
template<class T, class Tm>
void L2L(int childBegin, int childEnd, const Vec4<T>& Xparent, const Vec4<T>* Xchildren,
         const SphericalLocalExpansion<Tm>& Lparent, SphericalLocalExpansion<Tm>* Lchildren)
{
    constexpr int          PP  = ExpansionOrder;
    const auto&            tab = SphericalTables<PP>::instance();
    const std::complex<Tm> I(0, 1);

    std::complex<Tm> Ynm_buf[4 * PP * PP];
    std::complex<Tm> YnmTheta_buf[4 * PP * PP];

    for (int c = childBegin; c < childEnd; ++c)
    {
        Vec3<T> dist{Xchildren[c][0] - Xparent[0], Xchildren[c][1] - Xparent[1], Xchildren[c][2] - Xparent[2]};
        T       rho, alpha, beta;
        cart2sph(rho, alpha, beta, dist);
        evalMultipole<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), Tm(rho), Tm(alpha), Tm(beta)); // positive beta

        for (int j = 0; j < PP; ++j)
        {
            for (int k = 0; k <= j; ++k)
            {
                int             jk  = j * j + j + k;
                int             jks = j * (j + 1) / 2 + k;
                std::complex<Tm> L(0, 0);

                for (int n = j; n < PP; ++n)
                {
                    // negative m range
                    for (int m = j + k - n; m < 0; ++m)
                    {
                        if (n - j >= std::abs(m - k))
                        {
                            int jnkm = (n - j) * (n - j) + n - j + m - k;
                            int nm   = n * n + n - m;
                            int nms  = n * (n + 1) / 2 - m;
                            L += std::conj(Lparent[nms]) * Ynm_buf[jnkm] *
                                 Tm(oddeven(k) * tab.Anm[jnkm] * tab.Anm[jk] / tab.Anm[nm]);
                        }
                    }
                    // non-negative m range
                    for (int m = 0; m <= n; ++m)
                    {
                        if (n - j >= std::abs(m - k))
                        {
                            int jnkm = (n - j) * (n - j) + n - j + m - k;
                            int nm   = n * n + n + m;
                            int nms  = n * (n + 1) / 2 + m;
                            L += Lparent[nms] * std::pow(I, Tm(m - k - std::abs(m - k))) * Ynm_buf[jnkm] *
                                 Tm(tab.Anm[jnkm] * tab.Anm[jk] / tab.Anm[nm]);
                        }
                    }
                }
                Lchildren[c][jks] += L;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L2P — Local to Particle
// ---------------------------------------------------------------------------

//! @brief Evaluate local expansion at a target particle position
template<class Ta, class Tc, class Tm>
Vec4<Ta> L2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center, const SphericalLocalExpansion<Tm>& local)
{
    constexpr int          PP  = ExpansionOrder;
    const auto&            tab = SphericalTables<PP>::instance();
    const std::complex<Ta> I(0, 1);

    std::complex<Ta> Ynm_buf[4 * PP * PP];
    std::complex<Ta> YnmTheta_buf[4 * PP * PP];

    Vec3<Tc> dist{target[0] - center[0], target[1] - center[1], target[2] - center[2]};
    Ta       r, theta, phi;
    cart2sph(r, theta, phi, dist);
    evalMultipole<PP>(Ynm_buf, YnmTheta_buf, tab.prefactor.data(), r, theta, phi); // regular harmonics, positive phi

    Ta       potential = 0;
    Vec3<Ta> spherical{0, 0, 0};

    for (int n = 0; n < PP; ++n)
    {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;

        potential += (std::complex<Ta>(local[nms]) * Ynm_buf[nm]).real();
        spherical[0] += (std::complex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n; // +n (not -(n+1))
        spherical[1] += (std::complex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();

        for (int m = 1; m <= n; ++m)
        {
            nm  = n * n + n + m;
            nms = n * (n + 1) / 2 + m;

            potential += Ta(2) * (std::complex<Ta>(local[nms]) * Ynm_buf[nm]).real();
            spherical[0] += Ta(2) * (std::complex<Ta>(local[nms]) * Ynm_buf[nm]).real() / r * n;
            spherical[1] += Ta(2) * (std::complex<Ta>(local[nms]) * YnmTheta_buf[nm]).real();
            spherical[2] += Ta(2) * (std::complex<Ta>(local[nms]) * Ynm_buf[nm] * I).real() * m;
        }
    }

    Vec3<Ta> cartesian = sph2cart(r, theta, phi, spherical);

    // Negate potential for gravity convention (same as M2P)
    return acc + Vec4<Ta>{-potential, cartesian[0], cartesian[1], cartesian[2]};
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

template<class T, class KeyType, class Th, class Tm>
void computeGravityFMM(const KeyType* prefixes, const TreeNodeIndex* childOffsets,
                        const TreeNodeIndex* internalToLeaf,
                        std::span<const TreeNodeIndex> leafToInternalMap,
                        std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                        const SphericalMultipole<T>* multipoles, const LocalIndex* layout,
                        TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                        const T* z, const Th* h, const Tm* m, const cstone::Box<T>& box, float theta, float G,
                        Th* ugrav, Th* ax, Th* ay, Th* az, T* ugravTot)
{
    TreeNodeIndex numNodes = levelRange.back();
    T             invTheta = T(1) / T(theta);

    // 1. Compute geometric node centers and sizes for the dual-traversal MAC
    std::vector<Vec3<T>> geoCenters(numNodes);
    std::vector<Vec3<T>> geoSizes(numNodes);
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        KeyType  prefix       = prefixes[i];
        KeyType  startKey     = cstone::decodePlaceholderBit(prefix);
        unsigned level        = cstone::decodePrefixLength(prefix) / 3;
        auto     nodeBox      = cstone::sfcIBox(cstone::sfcKey(startKey), level);
        auto [center, sz]     = cstone::centerAndSize<KeyType>(nodeBox, box);
        geoCenters[i]         = center;
        geoSizes[i]           = sz;
    }

    // 2. Allocate local expansions (zero-initialized)
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
    std::atomic<unsigned> cpuM2lCount{0}, cpuP2pCount{0};

    auto continuation = [&geoCenters, &geoSizes, invTheta](TreeNodeIndex a, TreeNodeIndex b) -> bool
    {
        Vec3<T> d     = cstone::minDistance(geoCenters[a], geoSizes[a], geoCenters[b], geoSizes[b]);
        T       dist2 = norm2(d);

        T lA        = T(2) * std::max({geoSizes[a][0], geoSizes[a][1], geoSizes[a][2]});
        T lB        = T(2) * std::max({geoSizes[b][0], geoSizes[b][1], geoSizes[b][2]});
        T threshold = std::max(lA, lB) * invTheta;

        return dist2 < threshold * threshold; // true = MAC fails, keep descending
    };

    auto m2lCallback = [&centers, &multipoles, &locals, &cpuM2lCount](TreeNodeIndex a, TreeNodeIndex b)
    {
        cpuM2lCount.fetch_add(1u, std::memory_order_relaxed);
        M2L(util::makeVec3(centers[a]), util::makeVec3(centers[b]), multipoles[b], locals[a]);
    };

    auto p2pCallback = [internalToLeaf, layout, x, y, z, h, m, &pax, &pay, &paz, &ppot,
                        firstTarget, &cpuP2pCount](TreeNodeIndex a, TreeNodeIndex b)
    {
        cpuP2pCount.fetch_add(1u, std::memory_order_relaxed);
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

    printf("[FMM CPU] M2L calls: %u, P2P calls: %u\n",
           cpuM2lCount.load(), cpuP2pCount.load());

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
