/*
 * Distributed FMM scaling benchmark
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief MPI-distributed FMM vs BH scaling benchmark
 *
 * @author Timo Schwab <tischwab@ethz.ch>
 *
 * Runs both Barnes-Hut (via MultipoleHolder::compute()) and FMM dual traversal
 * (via computeGravityFMMGpuDistributed()) across MPI ranks with timing and
 * accuracy comparison.
 */

#include <algorithm>
#include <chrono>
#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domain.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"

#include "fmm/cartesian_qpole_fmm.cuh"

using AccType = cstone::GpuTag;

using namespace ryoanji;

template<class T, class KeyType>
void fmmScalingTest(int thisRank, int numRanks, size_t numParticlesGlobal)
{
    using MultipoleType            = CartesianQuadrupole<T>;
    size_t   numParticles          = numParticlesGlobal / numRanks;
    unsigned bucketSizeFocus       = 64;
    unsigned numGlobalNodesPerRank = 100;
    unsigned bucketSizeGlobal =
        std::max(size_t(bucketSizeFocus), numParticlesGlobal / (numGlobalNodesPerRank * numRanks));
    T              G        = 1.0;
    float          theta    = 0.5;
    float          invTheta = 1.0f / theta;
    cstone::Box<T> box{-1, 1};

    double hmean = std::cbrt(30.0 / 0.523 / double(numParticlesGlobal) * box.lx() * box.ly() * box.lz());

    int                                                    seed = thisRank;
    cstone::RandomCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box, seed);

    std::vector<T> h(numParticles, hmean);
    std::vector<T> m(numParticles, 1.0f / float(numParticlesGlobal));

    cstone::Domain<KeyType, T, AccType> domain(thisRank, numRanks, bucketSizeGlobal, bucketSizeFocus, theta, box);

    cstone::DeviceVector<KeyType> d_keys = std::vector<KeyType>(numParticles);
    cstone::DeviceVector<T>       d_x    = coords.x();
    cstone::DeviceVector<T>       d_y    = coords.y();
    cstone::DeviceVector<T>       d_z    = coords.z();
    cstone::DeviceVector<T>       d_h    = h;
    cstone::DeviceVector<T>       d_m    = m;
    cstone::DeviceVector<T>       s1, s2, s3;

    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    auto                                                      octree   = focusTree.octreeViewAcc();

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    auto grp = multipoleHolder.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d_x), rawPtr(d_y),
                                                    rawPtr(d_z), rawPtr(d_h), domain.focusTree(),
                                                    domain.layout().data(), domain.box());
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(),
                            domain.focusTree(), domain.layout().data());

    // ===== BH benchmark: 1 warmup + 5 timed runs =====
    cstone::DeviceVector<T> d_ax(domain.nParticlesWithHalos());
    cstone::DeviceVector<T> d_ay(domain.nParticlesWithHalos());
    cstone::DeviceVector<T> d_az(domain.nParticlesWithHalos());

    auto zeroBuf = [&]()
    {
        checkGpuErrors(cudaMemset(rawPtr(d_ax), 0, domain.nParticlesWithHalos() * sizeof(T)));
        checkGpuErrors(cudaMemset(rawPtr(d_ay), 0, domain.nParticlesWithHalos() * sizeof(T)));
        checkGpuErrors(cudaMemset(rawPtr(d_az), 0, domain.nParticlesWithHalos() * sizeof(T)));
    };

    constexpr int nWarmup = 1;
    constexpr int nRuns   = 5;

    // BH warmup
    for (int i = 0; i < nWarmup; ++i)
    {
        zeroBuf();
        multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G, 0, box,
                                nullptr, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az));
    }

    // BH timed runs (CUDA events to match FMM timing methodology)
    cudaEvent_t bhStart, bhEnd;
    checkGpuErrors(cudaEventCreate(&bhStart));
    checkGpuErrors(cudaEventCreate(&bhEnd));

    std::vector<double> bhTimes(nRuns);
    for (int i = 0; i < nRuns; ++i)
    {
        zeroBuf();
        checkGpuErrors(cudaEventRecord(bhStart));
        multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G, 0, box,
                                nullptr, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az));
        checkGpuErrors(cudaEventRecord(bhEnd));
        checkGpuErrors(cudaDeviceSynchronize());
        float ms = 0;
        checkGpuErrors(cudaEventElapsedTime(&ms, bhStart, bhEnd));
        bhTimes[i] = ms;
    }

    checkGpuErrors(cudaEventDestroy(bhStart));
    checkGpuErrors(cudaEventDestroy(bhEnd));

    // Keep last BH accelerations for error comparison
    auto dlOwned = [&](cstone::DeviceVector<T>& dv)
    {
        std::vector<T> ret(domain.nParticles());
        memcpyD2H(dv.data() + domain.startIndex(), domain.nParticles(), ret.data());
        return ret;
    };
    std::vector<T> bhAxOwned = dlOwned(d_ax);
    std::vector<T> bhAyOwned = dlOwned(d_ay);
    std::vector<T> bhAzOwned = dlOwned(d_az);

    // ===== FMM benchmark: 1 warmup + 5 timed runs =====
    auto d_fmmMultipoles = reinterpret_cast<const fmm::CartesianMultipole<T>*>(multipoleHolder.deviceMultipoles());
    auto ltiSpan         = octree.leafToInternalSpan();
    std::span<const cstone::SourceCenterType<T>> centers = focusTree.expansionCentersAcc();

    // FMM warmup
    for (int i = 0; i < nWarmup; ++i)
    {
        zeroBuf();
        T fmmPot = 0;
        fmm::computeGravityFMMGpuDistributed<fmm::DirectionalMac>(
            octree.prefixes, octree.childOffsets, octree.internalToLeaf, ltiSpan.data(), domain.layout().data(),
            centers.data(), d_fmmMultipoles, octree.levelRangeSpan(), octree.numNodes, octree.numLeafNodes,
            domain.nParticlesWithHalos(), domain.startIndex(), domain.endIndex(), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
            rawPtr(d_h), rawPtr(d_m), box, G, invTheta, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az), &fmmPot);
    }

    // FMM timed runs
    std::vector<double> fmmTimes(nRuns);
    for (int i = 0; i < nRuns; ++i)
    {
        zeroBuf();
        T                fmmPot = 0;
        fmm::FmmGpuStats fmmStats;
        fmm::computeGravityFMMGpuDistributed<fmm::DirectionalMac>(
            octree.prefixes, octree.childOffsets, octree.internalToLeaf, ltiSpan.data(), domain.layout().data(),
            centers.data(), d_fmmMultipoles, octree.levelRangeSpan(), octree.numNodes, octree.numLeafNodes,
            domain.nParticlesWithHalos(), domain.startIndex(), domain.endIndex(), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
            rawPtr(d_h), rawPtr(d_m), box, G, invTheta, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az), &fmmPot, &fmmStats);
        fmmTimes[i] = fmmStats.msTotal();
    }

    // Keep last FMM accelerations for error comparison
    std::vector<T> fmmAxOwned = dlOwned(d_ax);
    std::vector<T> fmmAyOwned = dlOwned(d_ay);
    std::vector<T> fmmAzOwned = dlOwned(d_az);

    // ===== Compute FMM vs BH error =====
    std::vector<T> errors(domain.nParticles());
    for (LocalIndex i = 0; i < domain.nParticles(); ++i)
    {
        Vec3<T> ref   = {bhAxOwned[i], bhAyOwned[i], bhAzOwned[i]};
        Vec3<T> probe = {fmmAxOwned[i], fmmAyOwned[i], fmmAzOwned[i]};
        T       refN  = norm2(ref);
        errors[i]     = refN > 0 ? std::sqrt(norm2(ref - probe) / refN) : 0;
    }
    std::sort(errors.begin(), errors.end());
    double errP99 = errors[size_t(errors.size() * 0.99)];

    // ===== Sort timings for percentiles =====
    std::sort(bhTimes.begin(), bhTimes.end());
    std::sort(fmmTimes.begin(), fmmTimes.end());

    // p50 = v[2], p90 = v[4], p99 = v[4] for 5 samples
    double bhP50 = bhTimes[2], bhP90 = bhTimes[4];
    double fmmP50 = fmmTimes[2], fmmP90 = fmmTimes[4];

    // ===== Report =====
    for (int rank = 0; rank < numRanks; ++rank)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == thisRank)
        {
            if (thisRank == 0)
            {
                fprintf(stdout, "=== Distributed FMM vs BH Scaling ===\n");
                fprintf(stdout, "numRanks: %d, particlesPerRank: %zu, theta: %.2f\n", numRanks, numParticles,
                        double(theta));
            }
            fprintf(stdout, "--- rank %d ----------------\n", thisRank);
            fprintf(stdout, "numParticles, numHalos   : %d, %d\n", domain.nParticles(),
                    domain.nParticlesWithHalos() - domain.nParticles());
            fprintf(stdout, "BH  p50/p90              : %.3f / %.3f ms\n", bhP50, bhP90);
            fprintf(stdout, "FMM p50/p90              : %.3f / %.3f ms\n", fmmP50, fmmP90);
            fprintf(stdout, "Speedup (BH/FMM p50)     : %.2fx\n", bhP50 / fmmP50);
            fprintf(stdout, "Eerr p99 (FMM vs BH)     : %.6e\n", errP99);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numParticles = argc > 1 ? std::stoll(argv[1]) : 1000000ll;

    fmmScalingTest<double, uint64_t>(rank, numRanks, numParticles);

    MPI_Finalize();
}
