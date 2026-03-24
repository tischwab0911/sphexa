/*
 * Distributed FMM correctness test
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief MPI-distributed FMM correctness test: compare FMM dual traversal against direct sum
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Uses Domain::syncGrav() for tree construction and MultipoleHolder::upsweep()
 * for distributed P2M + M2M + MPI exchange. The pre-computed multipoles are
 * fed to computeGravityFMMGpuDistributed() via reinterpret_cast (identical
 * layout: fmm::CartesianMultipole<T> == ryoanji::CartesianQuadrupole<T> == util::array<T, 8>).
 */

#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"

#include "fmm/cartesian_qpole_fmm.cuh"

using namespace ryoanji;

template<class T, class KeyType>
static int multipoleHolderTest(int thisRank, int numRanks)
{
    using MultipoleType              = CartesianQuadrupole<T>;
    const LocalIndex numParticles    = (100000 / numRanks) * numRanks;
    unsigned         bucketSize      = numParticles / (100 * numRanks);
    unsigned         bucketSizeLocal = std::min(64u, bucketSize);
    float            theta           = 0.5;
    float            invTheta        = 1.0f / theta;
    T                G               = 1.0;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::fixed);
    int            numShells = 0;

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box);
    coords.adjustH(5, 10);

    std::vector<T> globalMasses(numParticles, 1.0 / numParticles);

    LocalIndex firstIndex = (numParticles * thisRank) / numRanks;
    LocalIndex lastIndex  = (numParticles * (thisRank + 1)) / numRanks;

    // extract a slice of the common pool
    std::vector<T>       x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T>       y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T>       z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T>       h(coords.h().begin() + firstIndex, coords.h().begin() + lastIndex);
    std::vector<T>       m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);
    std::vector<KeyType> h_keys(x.size());

    cstone::Domain<KeyType, T, cstone::GpuTag> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    cstone::DeviceVector<KeyType> d_keys = h_keys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h;
    cstone::DeviceVector<T>       d_m = m;
    cstone::DeviceVector<T>       s1;
    cstone::DeviceVector<T>       s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    h_keys.resize(domain.nParticles());
    memcpyD2H(d_keys.data() + domain.startIndex(), domain.nParticles(), h_keys.data());

    // Map to global indices (same as global_forces_gpu.cpp)
    LocalIndex firstGlobalIdx =
        std::lower_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.front()) -
        coords.particleKeys().begin();
    LocalIndex lastGlobalIdx =
        std::upper_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.back()) -
        coords.particleKeys().begin();

    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    auto                                                      octree   = focusTree.octreeViewAcc();

    // Distributed upsweep: P2M + M2M + MPI exchange
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data());

    // reinterpret_cast: ryoanji::CartesianQuadrupole<T> == fmm::CartesianMultipole<T> == util::array<T, 8>
    auto d_fmmMultipoles = reinterpret_cast<const fmm::CartesianMultipole<T>*>(multipoleHolder.deviceMultipoles());

    // Extract tree data for FMM
    auto ltiSpan = octree.leafToInternalSpan();
    std::span<const cstone::SourceCenterType<T>> centers = focusTree.expansionCentersAcc();

    // FMM accelerations
    cstone::DeviceVector<T> d_fmmAx(domain.nParticlesWithHalos(), 0);
    cstone::DeviceVector<T> d_fmmAy(domain.nParticlesWithHalos(), 0);
    cstone::DeviceVector<T> d_fmmAz(domain.nParticlesWithHalos(), 0);

    T                fmmPotential = 0;
    fmm::FmmGpuStats fmmStats;

    fmm::computeGravityFMMGpuDistributed<fmm::DirectionalMac>(
        octree.prefixes,
        octree.childOffsets,
        octree.internalToLeaf,
        ltiSpan.data(),
        domain.layout().data(),
        centers.data(),
        d_fmmMultipoles,
        octree.levelRangeSpan(),
        octree.numNodes, octree.numLeafNodes,
        domain.nParticlesWithHalos(),
        domain.startIndex(), domain.endIndex(),
        rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), rawPtr(d_m),
        box, G, invTheta,
        rawPtr(d_fmmAx), rawPtr(d_fmmAy), rawPtr(d_fmmAz),
        &fmmPotential, &fmmStats);

    // Download FMM accelerations for owned particles
    auto dl = [](auto* p1, auto* p2)
    {
        std::vector<std::remove_pointer_t<decltype(p1)>> ret(p2 - p1);
        memcpyD2H(p1, p2 - p1, ret.data());
        return ret;
    };

    std::vector<T> fmmAx = dl(d_fmmAx.data() + domain.startIndex(), d_fmmAx.data() + domain.endIndex());
    std::vector<T> fmmAy = dl(d_fmmAy.data() + domain.startIndex(), d_fmmAy.data() + domain.endIndex());
    std::vector<T> fmmAz = dl(d_fmmAz.data() + domain.startIndex(), d_fmmAz.data() + domain.endIndex());

    // Direct sum reference using the full global set
    cstone::DeviceVector<T> d_xref = coords.x(), d_yref = coords.y(), d_zref = coords.z();
    cstone::DeviceVector<T> d_mref = globalMasses;
    cstone::DeviceVector<T> d_href = coords.h();

    cstone::DeviceVector<T> d_potref(numParticles, 0);
    cstone::DeviceVector<T> d_axref(numParticles, 0), d_ayref(numParticles, 0), d_azref(numParticles, 0);

    directSum(firstGlobalIdx, lastGlobalIdx, numParticles, {box.lx(), box.ly(), box.lz()}, numShells, rawPtr(d_xref),
              rawPtr(d_yref), rawPtr(d_zref), rawPtr(d_mref), rawPtr(d_href), rawPtr(d_potref), rawPtr(d_axref),
              rawPtr(d_ayref), rawPtr(d_azref));

    std::vector<T> pRef  = dl(d_potref.data() + firstGlobalIdx, d_potref.data() + lastGlobalIdx);
    std::vector<T> axRef = dl(d_axref.data() + firstGlobalIdx, d_axref.data() + lastGlobalIdx);
    std::vector<T> ayRef = dl(d_ayref.data() + firstGlobalIdx, d_ayref.data() + lastGlobalIdx);
    std::vector<T> azRef = dl(d_azref.data() + firstGlobalIdx, d_azref.data() + lastGlobalIdx);

    // Compute FMM errors vs direct sum
    double         potentialSumRef = 0;
    std::vector<T> fmmErrors(fmmAx.size());
    for (size_t i = 0; i < fmmAx.size(); i++)
    {
        potentialSumRef += pRef[i];
        Vec3<T> ref   = {axRef[i], ayRef[i], azRef[i]};
        Vec3<T> probe = {fmmAx[i], fmmAy[i], fmmAz[i]};
        fmmErrors[i]  = std::sqrt(norm2(ref - probe) / norm2(ref));
    }
    potentialSumRef *= 0.5 * G;
    std::sort(fmmErrors.begin(), fmmErrors.end());

    double err1pc = fmmErrors[size_t(fmmErrors.size() * 0.99)];
    double errmax = fmmErrors.back();

    std::vector<double> firstPercentiles(numRanks), maxErrors(numRanks);
    MPI_Allgather(&err1pc, 1, MPI_DOUBLE, firstPercentiles.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&errmax, 1, MPI_DOUBLE, maxErrors.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    double fmmPotentialGlob, potentialSumRefGlob;
    double fmmPotDbl = double(fmmPotential);
    MPI_Allreduce(&fmmPotDbl, &fmmPotentialGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&potentialSumRef, &potentialSumRefGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double atol1pc = 5e-2;
    double atolmax = 5e-1;
    double ptol    = 1e-2;

    if (thisRank == 0)
    {
        for (int i = 0; i < numRanks; ++i)
        {
            std::cout << "rank " << i << " FMM p99 acc error " << firstPercentiles[i] << ", max acc error "
                      << maxErrors[i] << std::endl;
        }
        std::cout << "global reference potential " << potentialSumRefGlob << ", FMM global potential "
                  << fmmPotentialGlob << std::endl;
        std::cout << "FMM timing: traversal " << fmmStats.msTraversal << " ms, L2L " << fmmStats.msL2L
                  << " ms, L2P " << fmmStats.msL2P << " ms, total " << fmmStats.msTotal() << " ms" << std::endl;
    }

    bool passAcc1pc = *std::max_element(firstPercentiles.begin(), firstPercentiles.end()) < atol1pc;
    bool passAccMax = *std::max_element(maxErrors.begin(), maxErrors.end()) < atolmax;
    bool passPot    = std::abs((fmmPotentialGlob - potentialSumRefGlob) / potentialSumRefGlob) < ptol;

    bool pass = passAcc1pc && passAccMax && passPot;

    if (thisRank == 0)
    {
        std::string testResult = pass ? "PASS" : "FAIL";
        std::cout << "Test result: " << testResult << std::endl;
        if (!passAcc1pc) { std::cout << "  FAILED: p99 error exceeds " << atol1pc << std::endl; }
        if (!passAccMax) { std::cout << "  FAILED: max error exceeds " << atolmax << std::endl; }
        if (!passPot) { std::cout << "  FAILED: potential error exceeds " << ptol << std::endl; }
    }

    if (pass) { return EXIT_SUCCESS; }
    else { return EXIT_FAILURE; }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = multipoleHolderTest<double, uint64_t>(rank, numRanks);

    MPI_Finalize();

    return testResult;
}
