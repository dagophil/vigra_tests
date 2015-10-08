cmake_minimum_required(VERSION 2.6)
project(vigra_newrf_tests)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/config)

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-braced-scalar-init")

# VIGRA
find_package(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

# HDF5
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})

# Boost
find_package(Boost 1.47 REQUIRED)
find_package(Boost COMPONENTS system filesystem python REQUIRED)
add_definitions(-DWITH_BOOST)
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${Boost_PYTHON_INCLUDE_DIR}
)

# Python
find_package(PythonLibs REQUIRED)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

# Test executable
add_executable(tests
    tests.cxx
)
target_link_libraries(tests
    ${VIGRA_IMPEX_LIBRARY}
    ${HDF5_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_PYTHON_LIBRARIES}
    pthread
    armadillo
)

# Parametertest executable
add_executable(parametertest
    parametertest.cxx
)
target_link_libraries(parametertest
    ${VIGRA_IMPEX_LIBRARY}
    ${HDF5_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_PYTHON_LIBRARIES}
    pthread
)

# Run vigra installation before running the tests (I always forget that...)
add_custom_target(install_vigra
    COMMAND make -C /home/pschill/bld/vigra_newrf/ install > /dev/null
)
add_dependencies(tests install_vigra)
add_dependencies(parametertest install_vigra)
