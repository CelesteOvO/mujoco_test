cmake_minimum_required(VERSION 3.10)

project(simulate+_imgui)

file(GLOB IMGUI_HDR *.h ./backends/*.h)
file(GLOB IMGUI_SRC *.cpp ./backends/*.cpp)

add_library(${PROJECT_NAME} OBJECT ${IMGUI_HDR} ${IMGUI_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC .)
target_include_directories(${PROJECT_NAME} PUBLIC ./backends)

target_link_libraries(
        ${PROJECT_NAME}
        glfw
)