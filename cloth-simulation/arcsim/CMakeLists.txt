cmake_minimum_required(VERSION 3.20)

if (WIN32)
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        add_compile_definitions(_USE_MATH_DEFINES)
endif()

project(arcsim)

set(CMAKE_CXX_STANDARD 20)

option(NO_OPENGL "Disable OpenGL support" OFF)

include_directories(src)
file(GLOB_RECURSE SOURCE "src/*.cpp" "src/*.hpp")
add_executable(arcsim ${SOURCE})

if(POLICY CMP0167)
        cmake_policy(SET CMP0167 NEW)
endif()

if(NO_OPENGL)
        add_definitions(-DNO_OPENGL)
else()
        # OpenGL
        find_package(OpenGL REQUIRED)

        # GLFW (replace GLUT)
        find_package(glfw3 REQUIRED)
endif()

# embree
if(APPLE)
        find_package(embree 4 REQUIRED)
endif()

# jsoncpp
if (WIN32)
        find_package(jsoncpp CONFIG REQUIRED)
elseif(APPLE)
else()
        find_path(JSONCPP_INCLUDE_DIR json/json.h)
        find_library(JSONCPP_LIBRARY jsoncpp)
        find_package(TBB REQUIRED COMPONENTS tbb)
endif()

#libigl

add_subdirectory(lib/libigl)
include_directories(lib/libigl/include)

add_subdirectory(lib/cnpy)
include_directories(lib/cnpy)

# PNG
find_package(PNG REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)

# CHOLMOD
find_path(SUITESPARSE_INCLUDE_DIR cholmod.h)
find_library(CHOLMOD_LIB cholmod)
find_library(SUITESPARSECONFIG_LIB suitesparseconfig)

# ALGLIB
add_subdirectory(lib/alglib)

# Boost
find_package(Boost COMPONENTS system filesystem thread REQUIRED)

# OpenMP
#find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_definitions(-DEIGEN_USE_TBB)

if (APPLE)
        if(NOT NO_OPENGL)
                add_definitions(-DGL_SILENCE_DEPRECATION)
        endif()
#        add_definitions(-D_NEON)
        include_directories(
                ${OPENGL_INCLUDE_DIRS}
                ${PNG_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIR}
                ${Boost_INCLUDE_DIRS}
                ${EMBREE_INCLUDE_DIRS}
                /opt/homebrew/include
                /opt/homebrew/include/suitesparse
        )
        target_link_directories(
                arcsim
                PRIVATE
               /opt/homebrew/lib
        )
        if(NO_OPENGL)
                target_link_libraries(
                        arcsim
                        embree
                        ${PNG_LIBRARIES}
                        alglib
                        ${Boost_LIBRARIES}
                        jsoncpp
                        tbb
                        cholmod
                        cnpy
                )
        else()
                target_link_libraries(
                        arcsim
                        embree
                        ${OPENGL_LIBRARIES}
                        ${PNG_LIBRARIES}
                        alglib
                        ${Boost_LIBRARIES}
                        jsoncpp
                        tbb
                        glfw
                        cholmod
                        cnpy
                )
        endif()

endif()

# Embree 4 release package (https://github.com/RenderKit/embree/releases); it also ships the TBB library used here.
set(EMBREE_ROOT "$ENV{EMBREE_ROOT}" CACHE PATH "Root of the Embree 4 installation (contains include/ and lib/)")

if(UNIX AND NOT APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
        add_definitions(-D_AVX)
        include_directories(
                ${OPENGL_INCLUDE_DIRS}
                ${PNG_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIR}
                ${Boost_INCLUDE_DIRS}
                /usr/include/suitesparse/
                /usr/include/jsoncpp
                ${EMBREE_ROOT}/include
        )
        target_link_directories(
                arcsim
                PRIVATE
                ${EMBREE_ROOT}/lib
        )
        if(NO_OPENGL)
                target_link_libraries(
                        arcsim
                        jsoncpp
                        ${PNG_LIBRARIES}
                        alglib
                        ${Boost_LIBRARIES}
                        embree4
                        tbb
                        ${CHOLMOD_LIB}
                        cnpy
                )
        else()
                target_link_libraries(
                        arcsim
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES}
                        jsoncpp
                        ${PNG_LIBRARIES}
                        alglib
                        ${Boost_LIBRARIES}
                        glfw
                        embree4
                        tbb
                        ${CHOLMOD_LIB}
                        cnpy
                )
        endif()
endif()

if (WIN32)
        target_link_libraries(
                arcsim
                ${OPENGL_LIBRARIES}
                ${GLUT_LIBRARIES}
                JsonCpp::JsonCpp
                ${PNG_LIBRARIES}
                alglib
                ${Boost_LIBRARIES}
        )
endif()