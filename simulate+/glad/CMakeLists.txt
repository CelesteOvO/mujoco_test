cmake_minimum_required(VERSION 3.15)

project(simulate+_glad)

add_library(${PROJECT_NAME} OBJECT)
target_sources(${PROJECT_NAME} PRIVATE src/glad.c include/glad/glad.h include/KHR/khrplatform.h)
target_include_directories(${PROJECT_NAME} PUBLIC include)