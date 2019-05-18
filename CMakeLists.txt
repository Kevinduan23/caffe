cmake_minimum_required(VERSION 3.1.7)
if (POLICY CMP0046)
    cmake_policy(SET CMP0046 NEW)
endif ()
if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif ()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)
include(GNUInstallDirs)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

set(INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib") # 假设安装目录在编译目录的lib子目录内

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 确保链接库不在系统默认安装的目录上时更改到项目lib上
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${CMAKE_INSTALL_RPATH} isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${INSTALL_LIB_DIR}")
endif ("${isSystemDir}" STREQUAL "-1")

# ---[ Options
caffe_option(CPU_ONLY "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "3" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
caffe_option(USE_EXTRA "Use extra packages" ON)

if (USE_OPENMP)
    if (APPLE)
        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
            set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif ()
        LINK_DIRECTORIES("${CMAKE_INSTALL_PREFIX}/lib")
    endif ()
endif ()

# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "Build with hdf5" ON)

# ---[ Dependencies
include(cmake/Dependencies.cmake)
include(thirdparty/ThirdParties.cmake)

# ---[ Flags
if (UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()

caffe_set_caffe_link()

if (USE_libstdcpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    include_directories(/usr/include)
    message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif ()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})
include_directories(${Caffe_INCLUDE_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if (HAVE_CUDA)
    # pass include pathes to cuda_include_directories()
    set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
    list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
    cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

    # add definitions to nvcc flags directly
    set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
    list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
    list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
    enable_language(CUDA)
endif ()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if (BUILD_python)
    add_custom_target(pytest COMMAND ${CMAKE_PREFIX_PATH}/bin/python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
    add_dependencies(pytest pycaffe)
endif ()

# ---[ uninstall target
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
        IMMEDIATE @ONLY)

add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

