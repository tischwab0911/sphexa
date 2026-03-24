list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../ryoanji/cmake")
include(setup_GTest)

set(FMM_TEST_INCLUDE_DIRS
    ${CSTONE_DIR} ${CSTONE_TEST_DIR}
    ${PROJECT_SOURCE_DIR}/../ryoanji/src
    ${PROJECT_SOURCE_DIR}/src)

set(testname fmm_cpu_unit_tests)
add_executable(${testname} spherical_bh_vs_direct.cpp spherical_fmm_vs_direct.cpp cartesian_fmm_vs_direct.cpp test_main.cpp)
target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_include_directories(${testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ${testname} COMMAND ${testname})

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(gpu_testname fmm_gpu_unit_tests)
    add_executable(${gpu_testname}
        spherical_fmm_gpu_vs_cpu.cu
        spherical_kernels_gpu_vs_cpu.cu
        cartesian_qpole_fmm_gpu_vs_bh.cu
        cartesian_fmm_fcc_vs_direct.cu
        test_main.cpp)
    target_compile_options(${gpu_testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${gpu_testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
    target_link_libraries(${gpu_testname} PRIVATE
        cstone_gpu cstone_thrust_incl ryoanji_kernels
        GTest::gtest_main OpenMP::OpenMP_CXX)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${gpu_testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${gpu_testname} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${gpu_testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${gpu_testname} COMMAND ${gpu_testname})

    set(tune_testname fmm_tune_benchmark)
    add_executable(${tune_testname} tune_fmm.cu test_main.cpp)
    target_compile_options(${tune_testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${tune_testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
    target_link_libraries(${tune_testname} PRIVATE
        cstone_gpu cstone_thrust_incl
        GTest::gtest_main OpenMP::OpenMP_CXX)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${tune_testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${tune_testname} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${tune_testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${tune_testname} COMMAND ${tune_testname})

    set(cart_tune_testname fmm_cart_tune_benchmark)
    add_executable(${cart_tune_testname} tune_cartesian_fmm.cu test_main.cpp)
    target_compile_options(${cart_tune_testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${cart_tune_testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
    target_link_libraries(${cart_tune_testname} PRIVATE
        cstone_gpu cstone_thrust_incl ryoanji_kernels
        GTest::gtest_main OpenMP::OpenMP_CXX)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${cart_tune_testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${cart_tune_testname} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${cart_tune_testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${cart_tune_testname} COMMAND ${cart_tune_testname})

    set(profile_testname fmm_profile_benchmark)
    add_executable(${profile_testname} profile_cartesian_fmm.cu test_main.cpp)
    target_compile_options(${profile_testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${profile_testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
    target_link_libraries(${profile_testname} PRIVATE
        cstone_gpu cstone_thrust_incl ryoanji_kernels
        GTest::gtest_main OpenMP::OpenMP_CXX)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${profile_testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${profile_testname} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${profile_testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    set(scaling_testname fmm_scaling_benchmark)
    add_executable(${scaling_testname} scaling_fmm_vs_bh.cu test_main.cpp)
    target_compile_options(${scaling_testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${scaling_testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS})
    target_link_libraries(${scaling_testname} PRIVATE
        cstone_gpu cstone_thrust_incl ryoanji_kernels
        GTest::gtest_main OpenMP::OpenMP_CXX)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${scaling_testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${scaling_testname} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${scaling_testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Distributed FMM correctness test (MPI)
    set(testname fmm_distributed_correctness)
    add_executable(${testname} distributed_fmm_correctness.cu)
    target_include_directories(${testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${testname} PRIVATE cstone_gpu ryoanji ${MPI_CXX_LIBRARIES})
    target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${testname} PRIVATE CUDA::cudart)
    endif()
    if(CMAKE_HIP_COMPILER)
        target_link_libraries(${testname} PRIVATE hip::host)
        set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    # Distributed FMM scaling benchmark (MPI)
    set(testname fmm_distributed_scaling)
    add_executable(${testname} distributed_fmm_scaling.cu)
    target_include_directories(${testname} PRIVATE ${FMM_TEST_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${testname} PRIVATE cstone_gpu ryoanji ${MPI_CXX_LIBRARIES})
    target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
    if(CMAKE_CUDA_COMPILER)
        target_compile_options(${testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_link_libraries(${testname} PRIVATE CUDA::cudart)
    endif()
    if(CMAKE_HIP_COMPILER)
        target_link_libraries(${testname} PRIVATE hip::host)
        set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
    endif()
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/fmm/)
endif()
