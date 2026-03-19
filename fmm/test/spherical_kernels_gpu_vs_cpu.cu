/*
 * Spherical harmonic kernel-level GPU vs CPU tests
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#include <array>
#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "fmm/spherical_multipole.hpp"
#include "fmm/spherical_multipole.cuh"

namespace
{

using T = double;
using CpuMultipole = fmm::SphericalMultipole<T>;
using CpuLocal = fmm::SphericalLocalExpansion<T>;
using GpuMultipole = fmm::GpuSphericalMultipole<T>;
using GpuLocal = fmm::GpuSphericalLocalExpansion<T>;

constexpr int kNterm = fmm::Nterm<fmm::ExpansionOrder>;

void clear(CpuMultipole& x)
{
    for (auto& v : x) { v = {0.0, 0.0}; }
}

void clear(CpuLocal& x)
{
    for (auto& v : x) { v = {0.0, 0.0}; }
}

void expectComplexNear(const std::complex<T>& a, const thrust::complex<T>& b, T tol)
{
    EXPECT_NEAR(a.real(), b.real(), tol);
    EXPECT_NEAR(a.imag(), b.imag(), tol);
}

__global__ void p2mSingleKernel(const T* x, const T* y, const T* z, const T* m,
                                ryoanji::LocalIndex n, ryoanji::Vec4<T> center,
                                GpuMultipole* out, const double* prefactor)
{
    GpuMultipole mp;
    fmm::P2MGpu<1>(x, y, z, m, 0, n, center, mp, prefactor);
    out[0] = mp;
}

__global__ void m2mSingleKernel(ryoanji::Vec4<T> parentCenter, ryoanji::Vec4<T> childCenter,
                                const GpuMultipole* childMp, GpuMultipole* out,
                                const double* prefactor, const double* Anm)
{
    GpuMultipole mOut;
    for (auto& v : mOut) { v = fmm::GpuComplex<T>(0.0, 0.0); }

    fmm::M2MGpu(parentCenter, childCenter, childMp[0], mOut, prefactor, Anm);
    out[0] = mOut;
}

__global__ void m2lSingleKernel(ryoanji::Vec3<T> targetCenter, ryoanji::Vec3<T> sourceCenter,
                                const GpuMultipole* mp, GpuLocal* out,
                                const double* prefactor, const double* Anm,
                                const fmm::GpuComplex<double>* Cnm)
{
    for (auto& v : out[0]) { v = fmm::GpuComplex<T>(0.0, 0.0); }

    fmm::M2LGpu(targetCenter, sourceCenter, mp[0], &out[0], prefactor, Anm, Cnm);
}

__global__ void l2lSingleKernel(ryoanji::Vec4<T> parentCenter, ryoanji::Vec4<T> childCenter,
                                const GpuLocal* parentLocal, GpuLocal* childLocal,
                                const double* prefactor, const double* Anm)
{
    GpuLocal local;
    for (auto& v : local) { v = fmm::GpuComplex<T>(0.0, 0.0); }

    fmm::L2LGpu(parentCenter, childCenter, parentLocal[0], local, prefactor, Anm);
    childLocal[0] = local;
}

__global__ void l2pSingleKernel(ryoanji::Vec4<T> acc, ryoanji::Vec3<T> target, ryoanji::Vec3<T> center,
                                const GpuLocal* local, ryoanji::Vec4<T>* out, const double* prefactor)
{
    out[0] = fmm::L2PGpu(acc, target, center, local[0], prefactor);
}

} // namespace

TEST(SphericalKernelsGpu, P2M)
{
    std::vector<T> x{0.12, -0.21, 0.44, -0.33, 0.51, -0.08};
    std::vector<T> y{-0.41, 0.07, -0.16, 0.35, -0.22, 0.28};
    std::vector<T> z{0.31, -0.18, 0.09, 0.23, -0.27, -0.14};
    std::vector<T> m{0.7, 0.3, 1.1, 0.9, 0.6, 0.4};

    ryoanji::Vec4<T> center{0.05, -0.12, 0.08, 0.0};

    CpuMultipole cpu{};
    clear(cpu);
    fmm::P2M(x.data(), y.data(), z.data(), m.data(), 0, ryoanji::LocalIndex(x.size()), center, cpu);

    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    T *d_x, *d_y, *d_z, *d_m;
    GpuMultipole* d_out;

    checkGpuErrors(cudaMalloc(&d_x, x.size() * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_y, y.size() * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_z, z.size() * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_m, m.size() * sizeof(T)));
    checkGpuErrors(cudaMalloc(&d_out, sizeof(GpuMultipole)));

    checkGpuErrors(cudaMemcpy(d_x, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_y, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_z, z.data(), z.size() * sizeof(T), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(d_m, m.data(), m.size() * sizeof(T), cudaMemcpyHostToDevice));

    p2mSingleKernel<<<1, 1>>>(d_x, d_y, d_z, d_m, ryoanji::LocalIndex(x.size()), center, d_out, tables.prefactor);
    checkGpuErrors(cudaDeviceSynchronize());

    GpuMultipole gpu{};
    checkGpuErrors(cudaMemcpy(&gpu, d_out, sizeof(GpuMultipole), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNterm; ++i) { expectComplexNear(cpu[i], gpu[i], 1e-11); }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_m);
    cudaFree(d_out);
    fmm::freeSphericalTables(tables);
}

TEST(SphericalKernelsGpu, M2M)
{
    ryoanji::Vec4<T> parentCenter{0.1, -0.2, 0.15, 0.0};
    ryoanji::Vec4<T> childCenter{0.22, -0.14, 0.09, 0.0};

    CpuMultipole child{};
    for (int i = 0; i < kNterm; ++i)
    {
        child[i] = std::complex<T>(0.01 * (i + 1), -0.02 * (i + 1));
    }

    CpuMultipole cpuOut{};
    clear(cpuOut);
    fmm::M2M(0, 1, parentCenter, &childCenter, &child, cpuOut);

    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    GpuMultipole *d_child, *d_out;
    checkGpuErrors(cudaMalloc(&d_child, sizeof(GpuMultipole)));
    checkGpuErrors(cudaMalloc(&d_out, sizeof(GpuMultipole)));
    checkGpuErrors(cudaMemcpy(d_child, &child, sizeof(CpuMultipole), cudaMemcpyHostToDevice));

    m2mSingleKernel<<<1, 1>>>(parentCenter, childCenter, d_child, d_out, tables.prefactor, tables.Anm);
    checkGpuErrors(cudaDeviceSynchronize());

    GpuMultipole gpuOut{};
    checkGpuErrors(cudaMemcpy(&gpuOut, d_out, sizeof(GpuMultipole), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNterm; ++i) { expectComplexNear(cpuOut[i], gpuOut[i], 1e-11); }

    cudaFree(d_child);
    cudaFree(d_out);
    fmm::freeSphericalTables(tables);
}

TEST(SphericalKernelsGpu, M2L)
{
    ryoanji::Vec3<T> targetCenter{0.17, -0.08, 0.24};
    ryoanji::Vec3<T> sourceCenter{-0.11, 0.13, -0.19};

    CpuMultipole multipole{};
    for (int i = 0; i < kNterm; ++i)
    {
        multipole[i] = std::complex<T>(-0.015 * (i + 1), 0.012 * (i + 1));
    }

    CpuLocal cpuLocal{};
    clear(cpuLocal);
    fmm::M2L(targetCenter, sourceCenter, multipole, cpuLocal);

    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    GpuMultipole* d_multipole;
    GpuLocal* d_local;
    checkGpuErrors(cudaMalloc(&d_multipole, sizeof(GpuMultipole)));
    checkGpuErrors(cudaMalloc(&d_local, sizeof(GpuLocal)));
    checkGpuErrors(cudaMemcpy(d_multipole, &multipole, sizeof(CpuMultipole), cudaMemcpyHostToDevice));

    m2lSingleKernel<<<1, 1>>>(targetCenter, sourceCenter, d_multipole, d_local, tables.prefactor, tables.Anm,
                              tables.Cnm);
    checkGpuErrors(cudaDeviceSynchronize());

    GpuLocal gpuLocal{};
    checkGpuErrors(cudaMemcpy(&gpuLocal, d_local, sizeof(GpuLocal), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNterm; ++i) { expectComplexNear(cpuLocal[i], gpuLocal[i], 1e-10); }

    cudaFree(d_multipole);
    cudaFree(d_local);
    fmm::freeSphericalTables(tables);
}

TEST(SphericalKernelsGpu, L2L)
{
    ryoanji::Vec4<T> parentCenter{0.03, -0.05, 0.11, 0.0};
    ryoanji::Vec4<T> childCenter{0.08, 0.01, 0.18, 0.0};

    CpuLocal parentLocal{};
    for (int i = 0; i < kNterm; ++i)
    {
        parentLocal[i] = std::complex<T>(0.02 * (i + 1), 0.013 * (i + 1));
    }

    std::array<ryoanji::Vec4<T>, 1> children{childCenter};
    std::array<CpuLocal, 1> cpuChild{};
    clear(cpuChild[0]);
    fmm::L2L(0, 1, parentCenter, children.data(), parentLocal, cpuChild.data());

    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    GpuLocal *d_parent, *d_child;
    checkGpuErrors(cudaMalloc(&d_parent, sizeof(GpuLocal)));
    checkGpuErrors(cudaMalloc(&d_child, sizeof(GpuLocal)));
    checkGpuErrors(cudaMemcpy(d_parent, &parentLocal, sizeof(CpuLocal), cudaMemcpyHostToDevice));

    l2lSingleKernel<<<1, 1>>>(parentCenter, childCenter, d_parent, d_child, tables.prefactor, tables.Anm);
    checkGpuErrors(cudaDeviceSynchronize());

    GpuLocal gpuChild{};
    checkGpuErrors(cudaMemcpy(&gpuChild, d_child, sizeof(GpuLocal), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNterm; ++i) { expectComplexNear(cpuChild[0][i], gpuChild[i], 1e-10); }

    cudaFree(d_parent);
    cudaFree(d_child);
    fmm::freeSphericalTables(tables);
}

TEST(SphericalKernelsGpu, L2P)
{
    ryoanji::Vec3<T> target{0.21, -0.14, 0.09};
    ryoanji::Vec3<T> center{0.02, -0.01, 0.03};
    ryoanji::Vec4<T> accIn{0.4, -0.2, 0.1, 0.05};

    CpuLocal local{};
    for (int i = 0; i < kNterm; ++i)
    {
        local[i] = std::complex<T>(-0.011 * (i + 1), 0.017 * (i + 1));
    }

    ryoanji::Vec4<T> cpuOut = fmm::L2P(accIn, target, center, local);

    fmm::GpuSphericalTables tables = fmm::uploadSphericalTables();

    GpuLocal* d_local;
    ryoanji::Vec4<T>* d_out;
    checkGpuErrors(cudaMalloc(&d_local, sizeof(GpuLocal)));
    checkGpuErrors(cudaMalloc(&d_out, sizeof(ryoanji::Vec4<T>)));
    checkGpuErrors(cudaMemcpy(d_local, &local, sizeof(CpuLocal), cudaMemcpyHostToDevice));

    l2pSingleKernel<<<1, 1>>>(accIn, target, center, d_local, d_out, tables.prefactor);
    checkGpuErrors(cudaDeviceSynchronize());

    ryoanji::Vec4<T> gpuOut{};
    checkGpuErrors(cudaMemcpy(&gpuOut, d_out, sizeof(ryoanji::Vec4<T>), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) { EXPECT_NEAR(cpuOut[i], gpuOut[i], 1e-10); }

    cudaFree(d_local);
    cudaFree(d_out);
    fmm::freeSphericalTables(tables);
}
