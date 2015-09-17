cmake_minimum_required(VERSION 2.6)
project(vigra_newrf_tests)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/config)

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# VIGRA
find_package(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

# HDF5
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})

# Test executable
add_executable(tests
    tests.cxx
)

target_link_libraries(tests
    ${VIGRA_IMPEX_LIBRARY}
    ${HDF5_LIBRARIES}
#    ${CMAKE_THREAD_LIBS_INIT}
)

# Run vigra installation before running the tests (I always forget that...)
add_custom_target(install_vigra
    COMMAND make -C /home/philip/bld/vigra_newrf/ install > /dev/null
)
add_dependencies(tests install_vigra)

