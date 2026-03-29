/*
 * Cartesian quadrupole FMM kernels
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Hardcoded Cartesian quadrupole P2M, M2M, M2L, L2L, L2P kernels + FMM driver
 *
 * Extends the ryoanji::CartesianQuadrupole Barnes-Hut implementation with
 * M2L, L2L, L2P kernels for a full FMM. Multipole layout matches
 * ryoanji::CartesianQuadrupole (trace-free with factor-of-3 convention).
 *
 * Pure-arithmetic kernel functions are annotated HOST_DEVICE_FUN so they
 * compile as __device__ in .cu translation units (used by cartesian_qpole_fmm.cuh).
 */

#pragma once

#include <cmath>
#include <vector>

#include "cstone/util/array.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/traversal.hpp"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/kernel.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"

#include "fmm_types.hpp"

namespace fmm
{

using ryoanji::Vec3;
using ryoanji::Vec4;
using ryoanji::LocalIndex;
using ryoanji::TreeNodeIndex;
using ryoanji::Cqi;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

//! @brief Multipole type alias — use ryoanji::CartesianQuadrupole directly
template<class T>
using CartesianMultipole = ryoanji::CartesianQuadrupole<T>;

//! @brief Local expansion type: {pot, gx, gy, gz, txx, txy, txz, tyy, tyz, tzz}
//! Stores Taylor coefficients: phi(x) ~ L0 + Li di + 1/2 Lij di dj
template<class T>
struct CartesianLocalExpansion : util::array<T, 10>
{
    using Base = util::array<T, 10>;
    using Base::operator[];
    using Base::operator=;
};

//! @brief Local expansion index names
struct Cli
{
    enum
    {
        pot = 0,
        gx,
        gy,
        gz,
        txx,
        txy,
        txz,
        tyy,
        tyz,
        tzz
    };
};

// P2M, P2M_add, P2M_finalize, addQuadrupole, M2M — use ryoanji implementations directly
using ryoanji::P2M;
using ryoanji::P2M_add;
using ryoanji::P2M_finalize;
using ryoanji::addQuadrupole;
using ryoanji::M2M;

// ---------------------------------------------------------------------------
// M2L — Multipole to Local (core FMM kernel)
// ---------------------------------------------------------------------------

/*! @brief Translate a source multipole expansion into a target local expansion
 *
 * @param[in]     targetCenter  center of the target local expansion
 * @param[in]     sourceCenter  center of the source multipole
 * @param[in]     multipole     source multipole moments
 * @param[inout]  local         target local expansion (contribution added)
 *
 * Computes Taylor coefficients of the electrostatic potential phi_e = M/r + ...
 * expanded about targetCenter. L2P then negates for gravity convention.
 *
 * R = targetCenter - sourceCenter, r = |R|
 *
 * L0   += M/r + 1/2 * (R^T Q R) / r^5
 * Li   += -M Ri/r^3 + (QR)i/r^5 - 5/2 (R^T Q R) Ri/r^7
 * Lij  += M(3 Ri Rj - r^2 dij)/r^5 + Qij/r^5
 *         - 5[(QR)i Rj + (QR)j Ri]/r^7 - 5/2 (R^T Q R) dij/r^7
 *         + 35/2 (R^T Q R) Ri Rj/r^9
 */
template<class T>
HOST_DEVICE_FUN void M2L(const Vec3<T>& targetCenter, const Vec3<T>& sourceCenter,
         const CartesianMultipole<T>& multipole, CartesianLocalExpansion<T>& local)
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

    // Q*R (matrix-vector product)
    T QRx = multipole[Cqi::qxx] * Rx + multipole[Cqi::qxy] * Ry + multipole[Cqi::qxz] * Rz;
    T QRy = multipole[Cqi::qxy] * Rx + multipole[Cqi::qyy] * Ry + multipole[Cqi::qyz] * Rz;
    T QRz = multipole[Cqi::qxz] * Rx + multipole[Cqi::qyz] * Ry + multipole[Cqi::qzz] * Rz;

    // R^T Q R (scalar)
    T rQr = Rx * QRx + Ry * QRy + Rz * QRz;

    // L0: potential
    local[Cli::pot] += M * r_minus1 + T(0.5) * rQr * r_minus5;

    // Li: gradient (= gravitational acceleration at center)
    local[Cli::gx] += -M * Rx * r_minus3 + QRx * r_minus5 - T(2.5) * rQr * Rx * r_minus7;
    local[Cli::gy] += -M * Ry * r_minus3 + QRy * r_minus5 - T(2.5) * rQr * Ry * r_minus7;
    local[Cli::gz] += -M * Rz * r_minus3 + QRz * r_minus5 - T(2.5) * rQr * Rz * r_minus7;

    // Lij: Hessian (tidal tensor)
    T mono5  = M * r_minus5;
    T rQr_r7 = rQr * r_minus7;
    T rQr_r9 = rQr * r_minus9;

    local[Cli::txx] += mono5 * (T(3) * Rx * Rx - r2) + multipole[Cqi::qxx] * r_minus5 -
                        T(10) * QRx * Rx * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Rx * Rx;

    local[Cli::txy] += mono5 * T(3) * Rx * Ry + multipole[Cqi::qxy] * r_minus5 -
                        T(5) * (QRx * Ry + QRy * Rx) * r_minus7 + T(17.5) * rQr_r9 * Rx * Ry;

    local[Cli::txz] += mono5 * T(3) * Rx * Rz + multipole[Cqi::qxz] * r_minus5 -
                        T(5) * (QRx * Rz + QRz * Rx) * r_minus7 + T(17.5) * rQr_r9 * Rx * Rz;

    local[Cli::tyy] += mono5 * (T(3) * Ry * Ry - r2) + multipole[Cqi::qyy] * r_minus5 -
                        T(10) * QRy * Ry * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Ry * Ry;

    local[Cli::tyz] += mono5 * T(3) * Ry * Rz + multipole[Cqi::qyz] * r_minus5 -
                        T(5) * (QRy * Rz + QRz * Ry) * r_minus7 + T(17.5) * rQr_r9 * Ry * Rz;

    local[Cli::tzz] += mono5 * (T(3) * Rz * Rz - r2) + multipole[Cqi::qzz] * r_minus5 -
                        T(10) * QRz * Rz * r_minus7 - T(2.5) * rQr_r7 + T(17.5) * rQr_r9 * Rz * Rz;
}

// ---------------------------------------------------------------------------
// L2L — Local to Local (Taylor shift from parent to children)
// ---------------------------------------------------------------------------

/*! @brief Shift a parent local expansion to its children
 *
 * For displacement d = childCenter - parentCenter:
 *   L'_0  = L_0 + L_i d_i + 1/2 L_ij d_i d_j
 *   L'_i  = L_i + L_ij d_j
 *   L'_ij = L_ij   (unchanged at quadrupole order)
 */
template<class T, class Tm>
HOST_DEVICE_FUN void L2L(int childBegin, int childEnd, const Vec4<T>& Xparent, const Vec4<T>* Xchildren,
         const CartesianLocalExpansion<Tm>& Lparent, CartesianLocalExpansion<Tm>* Lchildren)
{
    for (int c = childBegin; c < childEnd; ++c)
    {
        T dx = Xchildren[c][0] - Xparent[0];
        T dy = Xchildren[c][1] - Xparent[1];
        T dz = Xchildren[c][2] - Xparent[2];

        // L'_0 = L_0 + L_i d_i + 1/2 L_ij d_i d_j
        Lchildren[c][Cli::pot] += Lparent[Cli::pot] + Lparent[Cli::gx] * dx + Lparent[Cli::gy] * dy +
                                  Lparent[Cli::gz] * dz +
                                  Tm(0.5) * (Lparent[Cli::txx] * dx * dx + Lparent[Cli::tyy] * dy * dy +
                                             Lparent[Cli::tzz] * dz * dz) +
                                  Lparent[Cli::txy] * dx * dy + Lparent[Cli::txz] * dx * dz +
                                  Lparent[Cli::tyz] * dy * dz;

        // L'_i = L_i + L_ij d_j
        Lchildren[c][Cli::gx] +=
            Lparent[Cli::gx] + Lparent[Cli::txx] * dx + Lparent[Cli::txy] * dy + Lparent[Cli::txz] * dz;
        Lchildren[c][Cli::gy] +=
            Lparent[Cli::gy] + Lparent[Cli::txy] * dx + Lparent[Cli::tyy] * dy + Lparent[Cli::tyz] * dz;
        Lchildren[c][Cli::gz] +=
            Lparent[Cli::gz] + Lparent[Cli::txz] * dx + Lparent[Cli::tyz] * dy + Lparent[Cli::tzz] * dz;

        // L'_ij = L_ij (unchanged at this order)
        Lchildren[c][Cli::txx] += Lparent[Cli::txx];
        Lchildren[c][Cli::txy] += Lparent[Cli::txy];
        Lchildren[c][Cli::txz] += Lparent[Cli::txz];
        Lchildren[c][Cli::tyy] += Lparent[Cli::tyy];
        Lchildren[c][Cli::tyz] += Lparent[Cli::tyz];
        Lchildren[c][Cli::tzz] += Lparent[Cli::tzz];
    }
}

// ---------------------------------------------------------------------------
// L2P — Local to Particle
// ---------------------------------------------------------------------------

/*! @brief Evaluate local expansion at a target particle position
 *
 * For d = target - center:
 *   pot  = -(L_0 + L_i d_i + 1/2 L_ij d_i d_j)    [gravity: negate electrostatic potential]
 *   a_k  = L_k + L_kj d_j                           [a = grad(phi_e) = -grad(phi_grav)]
 */
template<class Ta, class Tc, class Tm>
HOST_DEVICE_FUN Vec4<Ta> L2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
             const CartesianLocalExpansion<Tm>& local)
{
    Ta dx = target[0] - center[0];
    Ta dy = target[1] - center[1];
    Ta dz = target[2] - center[2];

    // phi_e(x) ~ L_0 + L_i d_i + 1/2 L_ij d_i d_j
    Ta phi_e = local[Cli::pot] + local[Cli::gx] * dx + local[Cli::gy] * dy + local[Cli::gz] * dz +
               Ta(0.5) * (local[Cli::txx] * dx * dx + local[Cli::tyy] * dy * dy + local[Cli::tzz] * dz * dz) +
               local[Cli::txy] * dx * dy + local[Cli::txz] * dx * dz + local[Cli::tyz] * dy * dz;

    // grad(phi_e) ~ L_k + L_kj d_j
    Ta accel_x = local[Cli::gx] + local[Cli::txx] * dx + local[Cli::txy] * dy + local[Cli::txz] * dz;
    Ta accel_y = local[Cli::gy] + local[Cli::txy] * dx + local[Cli::tyy] * dy + local[Cli::tyz] * dz;
    Ta accel_z = local[Cli::gz] + local[Cli::txz] * dx + local[Cli::tyz] * dy + local[Cli::tzz] * dz;

    // gravity: pot = -phi_e, acceleration = grad(phi_e) = -grad(phi_grav)
    return acc + Vec4<Ta>{-phi_e, accel_x, accel_y, accel_z};
}

// ---------------------------------------------------------------------------
// L2L downsweep — propagate local expansions from root to leaves
// ---------------------------------------------------------------------------

template<class T>
void downsweepLocalExpansions(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* childOffsets,
                              const cstone::SourceCenterType<T>* centers, CartesianLocalExpansion<T>* locals)
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
                        const TreeNodeIndex* internalToLeaf, std::span<const TreeNodeIndex> leafToInternalMap,
                        std::span<const TreeNodeIndex> levelRange, const cstone::SourceCenterType<T>* centers,
                        const CartesianMultipole<T>* multipoles, const LocalIndex* layout,
                        TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T* x, const T* y,
                        const T* z, const Th* h, const Tm* m, const cstone::Box<T>& box, float G,
                        float invTheta, Th* ugrav, Th* ax, Th* ay, Th* az, T* ugravTot)
{
    TreeNodeIndex numNodes = levelRange.back();

    // 0. Compute geometric centers/sizes from SFC prefixes
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
    std::vector<CartesianLocalExpansion<T>> locals(numNodes);
    for (auto& L : locals)
        for (auto& v : L)
            v = T(0);

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
        const auto&   L      = locals[nodeIdx];
        Vec3<T>       center = util::makeVec3(centers[nodeIdx]);

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
