project(fmm CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
