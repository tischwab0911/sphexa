/*
 * Spherical harmonic Barnes-Hut test
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test spherical harmonic BH against direct sum
 *
 * Builds an octree, computes spherical harmonic multipoles (P=6),
 * runs a Barnes-Hut traversal, and compares against a direct sum reference.
 */

#include <chrono>
#include <numeric>

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

#include "fmm/spherical_multipole.hpp"

// ---------------------------------------------------------------------------
// Barnes-Hut traversal for SphericalMultipole (moved from spherical_multipole.hpp)
// ---------------------------------------------------------------------------

namespace fmm
{

// M2P — Multipole to Particle (moved from spherical_multipole.hpp, only used in BH test)
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

template<class T1, size_t N>
static auto computeCenterAndSize(const util::array<Vec4<T1>, N>& target)
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
static void computeGravityGroup(const util::array<Vec4<T1>, N>& target, const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents, const TreeNodeIndex* internalToLeaf,
                                const cstone::SourceCenterType<T1>* centers, const SphericalMultipole<T1>* multipoles,
                                const LocalIndex* layout, const T1* x, const T1* y, const T1* z, const Th* h,
                                const Tm* m, Vec4<T1>* acc)
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
                acc[k] = ryoanji::P2P(acc[k], util::makeVec3(target[k]), Vec3<T1>{x[s], y[s], z[s]}, m[s],
                                      Th(target[k][3]), h[s]);
            }
        }
    };

    cstone::singleTraversal(childOffsets, parents, descendOrM2P, leafP2P);
}

template<class T1, class T2, class Tm>
static void computeGravity(const TreeNodeIndex* childOffsets, const TreeNodeIndex* parents,
                           const TreeNodeIndex* internalToLeaf, const cstone::SourceCenterType<T1>* macSpheres,
                           const SphericalMultipole<T1>* multipoles, const LocalIndex* layout,
                           TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T1* x, const T1* y,
                           const T1* z, const T2* h, const Tm* m, const cstone::Box<T1>& box, float G, T2* ugrav,
                           T2* ax, T2* ay, T2* az, T1* ugravTot, int numShells = 0)
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

} // namespace fmm

using namespace cstone;

TEST(SphericalBH, VsDirectSum)
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
    int            numShells    = 0;
    LocalIndex     numParticles = 10000;

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

    // compute spherical multipoles (P2M + M2M upsweep)
    std::vector<MultipoleType> multipoles(octree.numNodes);
    ryoanji::computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    ryoanji::upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    // Barnes-Hut tree walk
    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    auto t0       = std::chrono::high_resolution_clock::now();
    T    egravTot = 0;
    fmm::computeGravity(octree.childOffsets.data(), octree.parents.data(), octree.internalToLeaf.data(), centers.data(),
                         multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h, masses.data(), box, G,
                         (T*)nullptr, ax.data(), ay.data(), az.data(), &egravTot, numShells);
    auto   t1      = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Spherical BH (P=" << fmm::ExpansionOrder << ") for " << numParticles << " particles: " << elapsed
              << " s" << std::endl;

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    ryoanji::directSum(x, y, z, h, masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, numShells, Ax.data(),
                       Ay.data(), Az.data(), potentialReference.data());
    t1      = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Direct sum: " << elapsed << " s" << std::endl;

    // check total gravitational energy
    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        refPotSum += potentialReference[i];
    }
    refPotSum *= 0.5;
    EXPECT_NEAR(std::abs(refPotSum - egravTot) / std::abs(refPotSum), 0, 1e-2);

    // relative acceleration errors
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> axi{ax[i], ay[i], az[i]}, Axi{Ax[i], Ay[i], Az[i]};
        delta[i] = std::sqrt(norm2(axi - Axi) / norm2(Axi));
    }

    std::sort(begin(delta), end(delta));

    std::cout.precision(10);
    std::cout << "min Error: " << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numParticles / 2] << std::endl;
    std::cout << "90th percentile: " << delta[LocalIndex(numParticles * 0.9)] << std::endl;
    std::cout << "99th percentile: " << delta[LocalIndex(numParticles * 0.99)] << std::endl;
    std::cout << "max Error: " << delta[numParticles - 1] << std::endl;

    EXPECT_LT(delta[LocalIndex(numParticles * 0.99)], 1e-3);
    EXPECT_LT(delta[numParticles - 1], 1e-2);
}
