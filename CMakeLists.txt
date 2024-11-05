cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
    celeste_test
    VERSION 0.1.0
)

enable_language(C)
enable_language(CXX)
if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

set(SAMPLE_STANDALONE OFF)
set(_INSTALL_SAMPLES ON)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SAMPLE_STANDALONE ON)
    # If standalone, do not install the samples.
    set(_INSTALL_SAMPLES OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(SAMPLE_STANDALONE)
    include(SampleOptions)
else()
    enforce_mujoco_macosx_min_version()
endif()
include(SampleDependencies)

set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MUJOCO_SAMPLE_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

if(MUJOCO_HARDEN)
    if(WIN32)
        set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,/DYNAMICBASE)
    else()
        set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${MUJOCO_SAMPLE_COMPILE_OPTIONS}" -fPIE)
        if(APPLE)
            set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,-pie)
        else()
            set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -pie)
        endif()
    endif()
endif()

#######################################################################
add_executable(hello courseCode/00-hello.cc)
target_compile_options(hello PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        hello
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(hello PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(ball courseCode/01-ball.cc)
target_compile_options(ball PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        ball
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(ball PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(pendulum courseCode/02-pendulum.cpp)
target_compile_options(pendulum PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        pendulum
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(pendulum PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(doublependulum courseCode/03-doublependulum.cpp)
target_compile_options(doublependulum PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        doublependulum
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(doublependulum PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(doublependulum_fsm courseCode/04-doublependulum_fsm.cpp)
target_compile_options(doublependulum_fsm PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        doublependulum_fsm
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(doublependulum_fsm PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(doublependulum_ik courseCode/05-doublependulum_ik.cpp)
target_compile_options(doublependulum_ik PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        doublependulum_ik
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(doublependulum_ik PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(dbpendulum_lqr courseCode/06-dbpendulum_lqr.cpp)
target_compile_options(dbpendulum_lqr PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        dbpendulum_lqr
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(dbpendulum_lqr PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(hybrid_pendulum courseCode/07-hybrid_pendulum.cpp)
target_compile_options(hybrid_pendulum PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        hybrid_pendulum
        mujoco::mujoco
        glfw
        Threads::Threads
)
target_link_options(hybrid_pendulum PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

set(PCL_DIR "C:\\Project\\PCL\\cmake")
find_package(PCL REQUIRED COMPONENTS visualization io common)
set(PCL_INCLUDE_DIRS "C:\\Project\\PCL\\include\\pcl-1.14")
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)

add_executable(RgbTest RGB/two_window.cpp RGB/mujoco_rgbd_camera.hpp RGB/mujoco_rgbd_camera.cpp)
target_include_directories(RgbTest PRIVATE ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
target_compile_options(RgbTest PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
        RgbTest
        mujoco::mujoco
        glfw
        Threads::Threads
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${VTK_LIBRARIES}
)

 target_link_options(RgbTest PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_subdirectory(simulate+)
#######################################################################

if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
    ######################################
    embed_in_bundle(ball simulate)
    embed_in_bundle(pendulum simulate)
    embed_in_bundle(doublependulum simulate)
    embed_in_bundle(doublependulum_fsm simulate)
    embed_in_bundle(doublependulum_ik simulate)
    embed_in_bundle(dbpendulum_lqr simulate)
    embed_in_bundle(hybrid_pendulum simulate)
    embed_in_bundle(RgbTest simulate)
    ######################################
endif()

# Do not install if macOS Bundles are created as RPATH is managed manually there.
if(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
    set(_INSTALL_SAMPLES OFF)
endif()

if(_INSTALL_SAMPLES)

    include(TargetAddRpath)

    # Add support to RPATH for the samples.
    target_add_rpath(
            TARGETS
            ######################################
            basic
            pendulum
            doublependulum
            doublependulum_fsm
            doublependulum_ik
            dbpendulum_lqr
            hybrid_pendulum
            RgbTest
            ######################################
            INSTALL_DIRECTORY
            "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
            LIB_DIRS
            "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
            DEPENDS
            MUJOCO_ENABLE_RPATH
    )

    install(
            TARGETS
            ######################################
            basic
            pendulum
            doublependulum
            doublependulum_fsm
            doublependulum_ik
            dbpendulum_lqr
            hybrid_pendulum
            RgbTest
            ######################################
            EXPORT ${PROJECT_NAME}
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
            BUNDLE DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT samples
    )

    if(NOT MUJOCO_SAMPLES_USE_SYSTEM_GLFW)
        # We downloaded GLFW. Depending if it is a static or shared LIBRARY we might
        # need to install it.
        get_target_property(MJ_GLFW_LIBRARY_TYPE glfw TYPE)
        if(MJ_GLFW_LIBRARY_TYPE STREQUAL SHARED_LIBRARY)
            install(
                    TARGETS glfw
                    EXPORT ${PROJECT_NAME}
                    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT samples
                    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
                    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT samples
                    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT samples
            )
        endif()
    endif()
endif()