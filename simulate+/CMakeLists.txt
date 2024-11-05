cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
        mujoco_simulate+
)

enable_language(C)
enable_language(CXX)
if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

set(PCL_DIR "C:\\Project\\PCL\\cmake")
find_package(PCL REQUIRED COMPONENTS visualization io common)
set(PCL_INCLUDE_DIRS "C:\\Project\\PCL\\include\\pcl-1.14")
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)
find_package(boost REQUIRED)


add_subdirectory(glad)
add_subdirectory(imgui)

add_library(libsimulate+ STATIC $<TARGET_OBJECTS:platform_ui_adapter>)
set_target_properties(libsimulate+ PROPERTIES OUTPUT_NAME simulate+)
add_library(mujoco::libsimulate+ ALIAS libsimulate+)

target_sources(
        libsimulate+
        PUBLIC simulate.h
        PRIVATE simulate.cc array_safety.h
)
target_include_directories(libsimulate+ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} )
target_compile_options(libsimulate+ PRIVATE ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
target_link_libraries(libsimulate+ PUBLIC lodepng mujoco::platform_ui_adapter mujoco::mujoco simulate+_imgui simulate+_glad
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${VTK_LIBRARIES}
        Boost::boost)
target_link_options(libsimulate+ PRIVATE ${MUJOCO_SIMULATE_LINK_OPTIONS})

add_executable(simulate+ main.cpp)

target_link_libraries(
        simulate+
        libsimulate+
        mujoco::mujoco
        glfw
        lodepng
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${VTK_LIBRARIES}
        Boost::boost
)

# 添加 Boost 的目录，如果找不到抛出错误
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Make sure Boost is installed and CMake can find it.")
endif()

# 如果找不到 VTK，抛出错误
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found. Make sure VTK is installed and CMake can find it.")
endif()